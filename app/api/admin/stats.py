"""Admin dashboard stats — counts + recent activity."""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...core.db import get_db
from ...core.deps import require_admin
from ...models import Analysis, AnalysisStatus, Message, Notification, User

router = APIRouter(prefix="/stats")


class StatusCount(BaseModel):
    queued: int
    processing: int
    completed: int
    failed: int


class StatsOut(BaseModel):
    users_total: int
    users_active: int
    admins: int
    analyses_total: int
    analyses_by_status: StatusCount
    notifications_total: int
    messages_total: int


class ActivityRow(BaseModel):
    kind: Literal["analysis", "user", "message"]
    title: str
    detail: str | None
    at: datetime


@router.get("", response_model=StatsOut)
def stats(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> StatsOut:
    users_total = db.query(func.count(User.id)).scalar() or 0
    users_active = (
        db.query(func.count(User.id)).filter(User.is_active.is_(True)).scalar() or 0
    )
    admins = db.query(func.count(User.id)).filter(User.role == "admin").scalar() or 0

    by_status = dict(
        db.query(Analysis.status, func.count(Analysis.id))
        .group_by(Analysis.status)
        .all()
    )

    def _count(s: AnalysisStatus) -> int:
        return int(by_status.get(s, 0) or 0)

    analyses_total = sum(int(v or 0) for v in by_status.values())
    notifications_total = db.query(func.count(Notification.id)).scalar() or 0
    messages_total = db.query(func.count(Message.id)).scalar() or 0

    return StatsOut(
        users_total=users_total,
        users_active=users_active,
        admins=admins,
        analyses_total=analyses_total,
        analyses_by_status=StatusCount(
            queued=_count(AnalysisStatus.queued),
            processing=_count(AnalysisStatus.processing),
            completed=_count(AnalysisStatus.completed),
            failed=_count(AnalysisStatus.failed),
        ),
        notifications_total=notifications_total,
        messages_total=messages_total,
    )


@router.get("/activity", response_model=list[ActivityRow])
def recent_activity(
    limit: int = 15,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> list[ActivityRow]:
    rows: list[ActivityRow] = []

    analyses = (
        db.query(Analysis, User)
        .join(User, Analysis.user_id == User.id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .all()
    )
    for a, u in analyses:
        rows.append(
            ActivityRow(
                kind="analysis",
                title=f"{u.display_name} • {a.source_name}",
                detail=a.status.value if hasattr(a.status, "value") else str(a.status),
                at=a.created_at,
            )
        )

    users = (
        db.query(User).order_by(User.created_at.desc()).limit(limit).all()
    )
    for u in users:
        rows.append(
            ActivityRow(
                kind="user",
                title=u.display_name,
                detail=u.email,
                at=u.created_at,
            )
        )

    rows.sort(key=lambda r: r.at, reverse=True)
    return rows[:limit]
