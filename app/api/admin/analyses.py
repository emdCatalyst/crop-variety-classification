"""Admin analysis management — list across all users + delete with disk cascade."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session, selectinload

from ...core.db import get_db
from ...core.deps import require_admin
from ...models import Analysis, AnalysisStatus, User
from ...services.cleanup import purge_analysis_artifacts

router = APIRouter(prefix="/analyses")


class AdminAnalysisOut(BaseModel):
    id: int
    source_name: str
    status: str
    error: str | None
    created_at: datetime
    updated_at: datetime
    user_id: int
    user_email: str
    user_display_name: str
    has_result: bool


@router.get("", response_model=list[AdminAnalysisOut])
def list_all_analyses(
    status_filter: str | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> list[AdminAnalysisOut]:
    q = (
        db.query(Analysis, User)
        .options(selectinload(Analysis.result))
        .join(User, Analysis.user_id == User.id)
    )
    if status_filter:
        try:
            q = q.filter(Analysis.status == AnalysisStatus(status_filter))
        except ValueError:
            raise HTTPException(status_code=400, detail="Unknown status filter")

    rows = q.order_by(Analysis.created_at.desc()).all()
    out: list[AdminAnalysisOut] = []
    for a, u in rows:
        out.append(
            AdminAnalysisOut(
                id=a.id,
                source_name=a.source_name,
                status=a.status.value if hasattr(a.status, "value") else str(a.status),
                error=a.error,
                created_at=a.created_at,
                updated_at=a.updated_at,
                user_id=u.id,
                user_email=u.email,
                user_display_name=u.display_name,
                has_result=a.result is not None,
            )
        )
    return out


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis_admin(
    analysis_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> Response:
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    purge_analysis_artifacts(a)
    db.delete(a)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
