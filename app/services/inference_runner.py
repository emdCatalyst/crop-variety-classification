"""Background worker that runs an analysis end-to-end.

Lazy-loads InferenceEngine on first call; subsequent jobs reuse the singleton.
Emits SSE events through core/events. Persists Result on success, marks Analysis
as failed (with error text) on exception.
"""
import asyncio
import logging
import threading
from pathlib import Path

from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.db import SessionLocal
from ..core.events import publish
from ..models import Analysis, AnalysisStatus, Result
from .health_advisor import HealthContext, assess as assess_health_ai
from .notifications import emit as emit_notification, user_channel as notifications_channel

log = logging.getLogger("agrovision.inference")

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from ml import InferenceEngine

                s = get_settings()
                _engine = InferenceEngine(
                    config={
                        "model_path": str(s.model_path),
                        "labels_csv": str(s.labels_csv),
                        "output_dir": str(s.maps_dir),
                    }
                )
    return _engine


async def _emit(channel: str, stage: str, **kwargs) -> None:
    await publish(channel, {"stage": stage, **kwargs})
    delay = get_settings().dev_stage_delay_s
    if delay > 0:
        await asyncio.sleep(delay)


async def run_analysis(analysis_id: int) -> None:
    channel = f"analysis:{analysis_id}"
    s = get_settings()
    db: Session = SessionLocal()
    try:
        analysis = db.get(Analysis, analysis_id)
        if not analysis:
            log.warning("analysis %s missing", analysis_id)
            return

        analysis.status = AnalysisStatus.processing
        db.commit()
        await _emit(channel, "loading", message="Loading GeoTIFFs")

        upload_dir = Path(analysis.upload_dir)
        tif_paths = sorted(str(p) for p in upload_dir.glob("*.tif")) + sorted(str(p) for p in upload_dir.glob("*.tiff"))
        if len(tif_paths) != s.num_timesteps:
            raise ValueError(f"Expected {s.num_timesteps} GeoTIFFs, found {len(tif_paths)}")

        await _emit(channel, "inferring", message="Running CNN+LSTM inference")
        engine = await asyncio.to_thread(_get_engine)

        artifacts = await asyncio.to_thread(
            engine.infer_tif_sequence,
            tif_paths,
            analysis.source_name,
            analysis.smooth,
            analysis.sigma,
            False,
            True,
        )

        await _emit(channel, "rendering", message="Persisting results")

        map_filename = Path(artifacts.map_png_path).name
        map_url = f"/static/maps/{map_filename}"

        assessment = await assess_health_ai(
            HealthContext(
                analysis_id=analysis.id,
                predicted_crop=artifacts.predicted_crop,
                class_distribution=artifacts.class_distribution,
                mean_confidence=float(artifacts.mean_confidence),
                mean_ndvi=float(artifacts.mean_ndvi),
                mean_ndre=float(artifacts.ndre_mean),
                mean_savi=float(artifacts.savi_mean),
                temporal_trend=float(artifacts.temporal_trend),
                heuristic_health_status=artifacts.health_status,
                heuristic_trajectory=artifacts.trajectory,
                heuristic_advice=artifacts.advice,
            )
        )
        log.info(
            "analysis %s health assessment from source=%s",
            analysis.id,
            assessment.source,
        )

        result = Result(
            analysis_id=analysis.id,
            predicted_crop=artifacts.predicted_crop,
            health_status=assessment.health_status,
            trajectory=assessment.trajectory,
            advice=assessment.advice,
            mean_confidence=float(artifacts.mean_confidence),
            mean_ndvi=float(artifacts.mean_ndvi),
            mean_ndre=float(artifacts.ndre_mean),
            mean_savi=float(artifacts.savi_mean),
            temporal_trend=float(artifacts.temporal_trend),
            map_png_path=artifacts.map_png_path,
            map_url=map_url,
            geotiff_path=artifacts.geotiff_path or None,
            class_distribution=artifacts.class_distribution,
        )
        db.add(result)
        analysis.status = AnalysisStatus.completed
        note = emit_notification(
            db,
            user_id=analysis.user_id,
            kind="analysis_completed",
            title=f"Analysis “{analysis.source_name}” completed",
            body=(
                f"Predicted crop: {artifacts.predicted_crop.replace('_', ' ')}. "
                f"Health: {assessment.health_status}."
            ),
            analysis_id=analysis.id,
        )
        db.commit()

        await publish(
            notifications_channel(analysis.user_id),
            {
                "id": note.id,
                "kind": note.kind,
                "title": note.title,
                "analysis_id": note.analysis_id,
            },
        )
        await _emit(channel, "done", message="Analysis complete", result_id=result.id)
    except Exception as exc:
        log.exception("analysis %s failed", analysis_id)
        note_payload: dict | None = None
        try:
            analysis = db.get(Analysis, analysis_id)
            if analysis:
                analysis.status = AnalysisStatus.failed
                analysis.error = str(exc)
                note = emit_notification(
                    db,
                    user_id=analysis.user_id,
                    kind="analysis_failed",
                    title=f"Analysis “{analysis.source_name}” failed",
                    body=str(exc)[:500],
                    analysis_id=analysis.id,
                )
                db.commit()
                note_payload = {
                    "id": note.id,
                    "kind": note.kind,
                    "title": note.title,
                    "analysis_id": note.analysis_id,
                    "user_id": analysis.user_id,
                }
        except Exception:
            pass
        if note_payload:
            await publish(notifications_channel(note_payload["user_id"]), note_payload)
        await _emit(channel, "failed", message=str(exc))
    finally:
        db.close()
