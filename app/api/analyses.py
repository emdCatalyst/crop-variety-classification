import re
import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..models import Analysis, AnalysisStatus, Image, User
from ..schemas.analysis import AnalysisDetailOut, AnalysisOut
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


@router.post("", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def create_analysis(
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
    if not analysis or analysis.user_id != user.id:
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
    if analysis.upload_dir:
        shutil.rmtree(analysis.upload_dir, ignore_errors=True)
    db.delete(analysis)
    db.commit()
