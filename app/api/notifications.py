"""In-app notifications: list / mark-as-read / delete + per-user SSE feed."""
import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..core.db import get_db
from ..core.deps import get_current_user, get_user_for_stream
from ..core.events import subscribe, unsubscribe
from ..models import Notification, User
from ..services.notifications import user_channel

router = APIRouter(prefix="/notifications", tags=["notifications"])


class NotificationOut(BaseModel):
    id: int
    kind: str
    title: str
    body: str
    analysis_id: int | None
    read_at: datetime | None
    created_at: datetime
    i18n_key: str | None = None
    i18n_params: dict | None = None

    class Config:
        from_attributes = True


class UnreadCountOut(BaseModel):
    unread: int


@router.get("", response_model=list[NotificationOut])
def list_notifications(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 50,
):
    return (
        db.query(Notification)
        .filter(Notification.user_id == user.id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
        .all()
    )


@router.get("/unread-count", response_model=UnreadCountOut)
def unread_count(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> UnreadCountOut:
    n = (
        db.query(Notification)
        .filter(Notification.user_id == user.id, Notification.read_at.is_(None))
        .count()
    )
    return UnreadCountOut(unread=n)


@router.post("/{note_id}/read", status_code=status.HTTP_204_NO_CONTENT)
def mark_read(
    note_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    note = db.get(Notification, note_id)
    if not note or note.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if note.read_at is None:
        note.read_at = datetime.now(timezone.utc)
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/read-all", status_code=status.HTTP_204_NO_CONTENT)
def mark_all_read(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Response:
    now = datetime.now(timezone.utc)
    (
        db.query(Notification)
        .filter(Notification.user_id == user.id, Notification.read_at.is_(None))
        .update({"read_at": now})
    )
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_notification(
    note_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    note = db.get(Notification, note_id)
    if not note or note.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    db.delete(note)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/stream")
async def notifications_stream(user: User = Depends(get_user_for_stream)):
    """Server-Sent Events feed of new notifications for the current user."""
    channel = user_channel(user.id)
    queue = await subscribe(channel)

    async def event_gen():
        try:
            yield {"event": "ready", "data": "{}"}
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
                    continue
                yield {"event": "notification", "data": json.dumps(event)}
        finally:
            await unsubscribe(channel, queue)

    return EventSourceResponse(event_gen())
