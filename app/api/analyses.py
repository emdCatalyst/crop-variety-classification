import re

from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
import shutil

from ..core.config import get_settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..core.rate_limit import limiter
from ..models import Analysis, AnalysisStatus, Image, User
from ..schemas.analysis import AnalysisDetailOut, AnalysisOut
from ..services.cleanup import purge_analysis_artifacts
from ..services.inference_runner import run_analysis

router = APIRouter(prefix="/analyses", tags=["analyses"])

_safe_re = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_filename(name: str) -> str:
    cleaned = _safe_re.sub("_", name).strip("._")
    return cleaned or "file.tif"


@router.get("", response_model=list[AnalysisOut])
def list_analyses(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Analysis]:
    return (
        db.query(Analysis)
        .filter(Analysis.user_id == user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )


class TimeseriesPoint(BaseModel):
    date: str
    count: int
    completed: int
    failed: int


@router.get("/stats/timeseries", response_model=list[TimeseriesPoint])
def analyses_timeseries(
    days: int = Query(default=30, ge=1, le=180),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[TimeseriesPoint]:
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)

    rows = (
        db.query(
            func.date(Analysis.created_at).label("d"),
            Analysis.status,
            func.count(Analysis.id),
        )
        .filter(Analysis.user_id == user.id, Analysis.created_at >= start_dt)
        .group_by("d", Analysis.status)
        .all()
    )

    buckets: dict[str, dict[str, int]] = {}
    for d_raw, st, cnt in rows:
        d = str(d_raw)
        bucket = buckets.setdefault(d, {"count": 0, "completed": 0, "failed": 0})
        bucket["count"] += int(cnt)
        st_val = st.value if hasattr(st, "value") else str(st)
        if st_val in ("completed", "failed"):
            bucket[st_val] += int(cnt)

    out: list[TimeseriesPoint] = []
    for i in range(days):
        d = start + timedelta(days=i)
        key = d.isoformat()
        b = buckets.get(key, {"count": 0, "completed": 0, "failed": 0})
        out.append(
            TimeseriesPoint(
                date=key,
                count=b["count"],
                completed=b["completed"],
                failed=b["failed"],
            )
        )
    return out


@router.post("", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("12/hour")
async def create_analysis(
    request: Request,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    source_name: str = Form("uploaded_sequence"),
    smooth: bool = Form(False),
    sigma: float = Form(3.0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Analysis:
    s = get_settings()
    if len(files) != s.num_timesteps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected exactly {s.num_timesteps} GeoTIFF files, got {len(files)}",
        )

    analysis = Analysis(
        user_id=user.id,
        source_name=source_name[:200],
        status=AnalysisStatus.queued,
        smooth=smooth,
        sigma=sigma,
        upload_dir="",
    )
    db.add(analysis)
    db.flush()

    upload_dir = s.upload_dir / str(user.id) / str(analysis.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    analysis.upload_dir = str(upload_dir)

    total_bytes = 0
    max_total = s.max_upload_mb * 1024 * 1024

    for idx, upload in enumerate(sorted(files, key=lambda f: f.filename or "")):
        filename = _safe_filename(upload.filename or f"t{idx:02d}.tif")
        target = upload_dir / f"{idx:02d}_{filename}"
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        size = target.stat().st_size
        total_bytes += size
        if total_bytes > max_total:
            shutil.rmtree(upload_dir, ignore_errors=True)
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Upload exceeds {s.max_upload_mb} MB limit",
            )
        db.add(
            Image(
                analysis_id=analysis.id,
                filename=filename,
                path=str(target),
                size_bytes=size,
                sequence_index=idx,
            )
        )

    db.commit()
    db.refresh(analysis)

    background.add_task(run_analysis, analysis.id)
    return analysis


@router.get("/{analysis_id}", response_model=AnalysisDetailOut)
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Analysis:
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if analysis.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return analysis


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    analysis = db.get(Analysis, analysis_id)
    if not analysis or analysis.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    purge_analysis_artifacts(analysis)
    db.delete(analysis)
    db.commit()
