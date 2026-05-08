"""User ↔ admin messaging.

Threads are 1:1, keyed by `min(uid):max(uid)`. Non-admin users may only message
the first admin (auto-resolved). Admins see all threads they participate in.
Attachments are limited to small JPEG/PNG/WEBP.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..core.config import get_settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..core.events import publish, subscribe, unsubscribe
from ..models import Message, User
from ..schemas.message import MessageOut, ThreadOut
from ..services.messages import (
    ALLOWED_MIME,
    MAX_ATTACHMENT_BYTES,
    attachment_dir,
    find_admin_recipient,
    preview,
    safe_attachment_name,
    thread_key,
    user_channel,
)

router = APIRouter(prefix="/messages", tags=["messages"])


class UnreadCountOut(BaseModel):
    unread: int


def _to_out(msg: Message, sender_name: str) -> MessageOut:
    return MessageOut(
        id=msg.id,
        sender_id=msg.sender_id,
        sender_name=sender_name,
        recipient_id=msg.recipient_id,
        body=msg.body,
        has_attachment=msg.attachment_path is not None,
        attachment_name=msg.attachment_name,
        attachment_mime=msg.attachment_mime,
        read_at=msg.read_at,
        created_at=msg.created_at,
        archived=msg.archived,
    )


def _resolve_other_user(
    db: Session, current: User, other_user_id: int | None
) -> User:
    if current.role == "admin":
        if other_user_id is None:
            raise HTTPException(status_code=400, detail="other_user_id is required for admins")
        other = db.get(User, other_user_id)
        if not other:
            raise HTTPException(status_code=404, detail="Recipient not found")
        return other
    # Non-admin: collapse to the first admin regardless of caller-provided id.
    admin = find_admin_recipient(db)
    if not admin:
        raise HTTPException(status_code=503, detail="No admin is available to message")
    return admin


@router.get("/threads", response_model=list[ThreadOut])
def list_threads(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[ThreadOut]:
    msgs = (
        db.query(Message)
        .filter(or_(Message.sender_id == user.id, Message.recipient_id == user.id))
        .order_by(Message.created_at.asc())
        .all()
    )
    if not msgs and user.role != "admin":
        admin = find_admin_recipient(db)
        if admin and admin.id != user.id:
            now = datetime.now(timezone.utc)
            return [
                ThreadOut(
                    thread_key=thread_key(user.id, admin.id),
                    other_user_id=admin.id,
                    other_user_name=admin.display_name,
                    other_user_role=admin.role,
                    last_body=None,
                    last_has_attachment=False,
                    last_at=now,
                    unread_count=0,
                    archived=False,
                )
            ]
        return []

    user_ids: set[int] = set()
    for m in msgs:
        user_ids.add(m.sender_id)
        user_ids.add(m.recipient_id)
    user_ids.discard(user.id)
    others = {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()}

    grouped: dict[str, list[Message]] = {}
    for m in msgs:
        grouped.setdefault(m.thread_key, []).append(m)

    out: list[ThreadOut] = []
    for key, items in grouped.items():
        last = items[-1]
        other_id = last.sender_id if last.sender_id != user.id else last.recipient_id
        other = others.get(other_id)
        if not other:
            continue
        unread = sum(1 for m in items if m.recipient_id == user.id and m.read_at is None)
        archived_thread = all(m.archived for m in items)
        out.append(
            ThreadOut(
                thread_key=key,
                other_user_id=other.id,
                other_user_name=other.display_name,
                other_user_role=other.role,
                last_body=preview(last.body, last.attachment_path is not None),
                last_has_attachment=last.attachment_path is not None,
                last_at=last.created_at,
                unread_count=unread,
                archived=archived_thread,
            )
        )
    out.sort(key=lambda t: t.last_at, reverse=True)
    return out


@router.get("", response_model=list[MessageOut])
def list_messages(
    with_user_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[MessageOut]:
    other = _resolve_other_user(db, user, with_user_id)
    key = thread_key(user.id, other.id)
    rows = (
        db.query(Message)
        .filter(Message.thread_key == key)
        .order_by(Message.created_at.asc())
        .all()
    )
    name_by_id = {user.id: user.display_name, other.id: other.display_name}
    return [_to_out(m, name_by_id.get(m.sender_id, "")) for m in rows]


@router.post("", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
async def send_message(
    body: str | None = Form(default=None),
    recipient_id: int | None = Form(default=None),
    attachment: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MessageOut:
    other = _resolve_other_user(db, user, recipient_id)
    if other.id == user.id:
        raise HTTPException(status_code=400, detail="Cannot message yourself")

    body_clean = (body or "").strip() or None
    has_attachment = attachment is not None and (attachment.filename or "").strip() != ""
    if not body_clean and not has_attachment:
        raise HTTPException(status_code=400, detail="Message must have a body or an attachment")

    key = thread_key(user.id, other.id)

    msg = Message(
        sender_id=user.id,
        recipient_id=other.id,
        thread_key=key,
        body=body_clean,
    )
    db.add(msg)
    db.flush()

    if has_attachment:
        assert attachment is not None
        mime = (attachment.content_type or "").lower()
        if mime not in ALLOWED_MIME:
            db.rollback()
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported attachment type. Allowed: {', '.join(sorted(ALLOWED_MIME))}",
            )
        s = get_settings()
        target_dir = attachment_dir(s.upload_dir, key)
        safe_name = safe_attachment_name(attachment.filename or "image")
        target = target_dir / f"{msg.id}_{safe_name}"
        size = 0
        with target.open("wb") as fh:
            while True:
                chunk = await attachment.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ATTACHMENT_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    db.rollback()
                    raise HTTPException(
                        status_code=413,
                        detail=f"Attachment exceeds {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB",
                    )
                fh.write(chunk)
        msg.attachment_path = str(target)
        msg.attachment_mime = mime
        msg.attachment_name = safe_name

    db.commit()
    db.refresh(msg)

    out = _to_out(msg, user.display_name)
    payload = {
        "id": msg.id,
        "thread_key": key,
        "sender_id": msg.sender_id,
        "sender_name": user.display_name,
        "recipient_id": msg.recipient_id,
        "preview": preview(msg.body, msg.attachment_path is not None),
        "created_at": msg.created_at.isoformat(),
    }
    await publish(user_channel(other.id), payload)
    await publish(user_channel(user.id), payload)
    return out


@router.get("/unread-count", response_model=UnreadCountOut)
def unread_count(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> UnreadCountOut:
    n = (
        db.query(Message)
        .filter(Message.recipient_id == user.id, Message.read_at.is_(None))
        .count()
    )
    return UnreadCountOut(unread=n)


@router.post("/read", status_code=status.HTTP_204_NO_CONTENT)
def mark_thread_read(
    with_user_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    other = _resolve_other_user(db, user, with_user_id)
    key = thread_key(user.id, other.id)
    now = datetime.now(timezone.utc)
    (
        db.query(Message)
        .filter(
            Message.thread_key == key,
            Message.recipient_id == user.id,
            Message.read_at.is_(None),
        )
        .update({"read_at": now}, synchronize_session=False)
    )
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{message_id}/archive", status_code=status.HTTP_204_NO_CONTENT)
def archive_message(
    message_id: int,
    archived: bool = Query(default=True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    msg = db.get(Message, message_id)
    if not msg or (msg.sender_id != user.id and msg.recipient_id != user.id):
        raise HTTPException(status_code=404, detail="Not found")
    msg.archived = archived
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{message_id}/attachment")
def download_attachment(
    message_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    msg = db.get(Message, message_id)
    if not msg or (msg.sender_id != user.id and msg.recipient_id != user.id):
        raise HTTPException(status_code=404, detail="Not found")
    if not msg.attachment_path:
        raise HTTPException(status_code=404, detail="No attachment")
    path = Path(msg.attachment_path)
    if not path.is_file():
        raise HTTPException(status_code=410, detail="Attachment file is gone")
    data = path.read_bytes()
    return Response(
        content=data,
        media_type=msg.attachment_mime or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{msg.attachment_name or path.name}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/stream")
async def messages_stream(user: User = Depends(get_current_user)):
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
                yield {"event": "message", "data": json.dumps(event, default=str)}
        finally:
            await unsubscribe(channel, queue)

    return EventSourceResponse(event_gen())
