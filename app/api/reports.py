"""Reports endpoints — history listing, farmer notes editing, PDF download."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session, selectinload

from ..core.db import get_db
from ..core.deps import get_current_user
from ..models import Analysis, AnalysisStatus, Result, User
from ..schemas.analysis import FarmerNotesIn, ReportListOut
from ..services.pdf import build_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=list[ReportListOut])
def list_reports(
    status_filter: str | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = (
        db.query(Analysis)
        .options(selectinload(Analysis.result))
        .filter(Analysis.user_id == user.id)
    )
    if status_filter:
        try:
            q = q.filter(Analysis.status == AnalysisStatus(status_filter))
        except ValueError:
            raise HTTPException(status_code=400, detail="Unknown status filter")
    rows = q.order_by(Analysis.created_at.desc()).all()
    out: list[ReportListOut] = []
    for a in rows:
        r = a.result
        out.append(
            ReportListOut(
                id=a.id,
                source_name=a.source_name,
                status=a.status.value if hasattr(a.status, "value") else str(a.status),
                created_at=a.created_at,
                predicted_crop=r.predicted_crop if r else None,
                health_status=r.health_status if r else None,
                farmer_notes=r.farmer_notes if r else None,
                observed_at=r.observed_at if r else None,
                has_result=r is not None,
            )
        )
    return out


@router.patch("/{analysis_id}/notes", status_code=status.HTTP_204_NO_CONTENT)
def update_notes(
    analysis_id: int,
    payload: FarmerNotesIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    analysis = db.get(Analysis, analysis_id)
    if not analysis or analysis.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not analysis.result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Notes can only be saved on completed analyses",
        )
    analysis.result.farmer_notes = (payload.farmer_notes or "").strip() or None
    if payload.observed_at:
        analysis.result.observed_at = payload.observed_at
    elif analysis.result.farmer_notes and not analysis.result.observed_at:
        analysis.result.observed_at = datetime.now(timezone.utc)
    elif not analysis.result.farmer_notes:
        analysis.result.observed_at = None
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{analysis_id}/pdf")
def download_pdf(
    analysis_id: int,
    lang: str = Query(default="en", pattern="^(en|fr|ar)$"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    analysis = (
        db.query(Analysis)
        .options(selectinload(Analysis.result))
        .filter(Analysis.id == analysis_id, Analysis.user_id == user.id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not analysis.result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Report is not ready until the analysis completes",
        )
    pdf_bytes = build_report(analysis, lang=lang)
    filename = f"agrovision-{analysis.id}-{lang}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
