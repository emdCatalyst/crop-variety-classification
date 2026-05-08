"""Admin notification broadcast — emit a notification to every active user."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.db import get_db
from ...core.deps import require_admin
from ...core.events import publish
from ...models import User
from ...services.notifications import emit, user_channel

router = APIRouter(prefix="/notifications")


class BroadcastIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)
    only_active: bool = True


class BroadcastOut(BaseModel):
    sent: int


@router.post("/broadcast", response_model=BroadcastOut, status_code=status.HTTP_201_CREATED)
async def broadcast(
    payload: BroadcastIn,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> BroadcastOut:
    title = payload.title.strip()
    body = payload.body.strip()
    if not title or not body:
        raise HTTPException(status_code=400, detail="title and body are required")

    q = db.query(User)
    if payload.only_active:
        q = q.filter(User.is_active.is_(True))
    targets = q.all()

    notes_payload: list[tuple[int, int, str, str]] = []
    for u in targets:
        note = emit(
            db,
            user_id=u.id,
            kind="broadcast",
            title=title[:200],
            body=body,
        )
        notes_payload.append((u.id, note.id, note.kind, note.title))
    db.commit()

    for user_id, note_id, kind, t in notes_payload:
        await publish(
            user_channel(user_id),
            {"id": note_id, "kind": kind, "title": t, "analysis_id": None},
        )

    return BroadcastOut(sent=len(targets))
