"""Helpers for the user ↔ admin messaging feature.

Conversations are 1:1, keyed by the sorted pair of user IDs (`thread_key`).
Image attachments are limited to small JPEG/PNG/WEBP and stored under
`<upload_dir>/messages/<thread_key>/<message_id>_<safe_name>`.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from ..models import Message, User

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB

_safe_re = re.compile(r"[^a-zA-Z0-9._-]+")


def thread_key(a: int, b: int) -> str:
    lo, hi = sorted((int(a), int(b)))
    return f"{lo}:{hi}"


def user_channel(user_id: int) -> str:
    return f"user:{user_id}:messages"


def find_admin_recipient(db: Session) -> User | None:
    """First active admin user, used as the default recipient for non-admins."""
    return (
        db.query(User)
        .filter(User.role == "admin", User.is_active.is_(True))
        .order_by(User.id.asc())
        .first()
    )


def safe_attachment_name(name: str) -> str:
    cleaned = _safe_re.sub("_", name).strip("._")
    return (cleaned or "file")[:180]


def attachment_dir(uploads_root: Path, key: str) -> Path:
    target = uploads_root / "messages" / key
    target.mkdir(parents=True, exist_ok=True)
    return target


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def active_conversation_id(db: Session, thread_key_str: str) -> str | None:
    """Return the conversation_id for the most-recent NON-archived message in
    this thread_key, or None if there's no live conversation (the previous
    one was archived, or no messages exist yet).
    """
    row = (
        db.query(Message.conversation_id)
        .filter(
            Message.thread_key == thread_key_str,
            Message.archived.is_(False),
        )
        .order_by(Message.created_at.desc())
        .first()
    )
    return row[0] if row else None


def preview(body: str | None, has_attachment: bool) -> str | None:
    if body and body.strip():
        snippet = body.strip().replace("\n", " ")
        return snippet[:140]
    if has_attachment:
        return "[attachment]"
    return None
