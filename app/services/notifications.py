"""Helpers for emitting in-app notifications."""
from typing import Any

from sqlalchemy.orm import Session

from ..models import Notification


def user_channel(user_id: int) -> str:
    """SSE channel name for a user's notification stream."""
    return f"user:{user_id}:notifications"


def emit(
    db: Session,
    *,
    user_id: int,
    kind: str,
    title: str,
    body: str,
    analysis_id: int | None = None,
    i18n_key: str | None = None,
    i18n_params: dict[str, Any] | None = None,
) -> Notification:
    note = Notification(
        user_id=user_id,
        kind=kind,
        title=title,
        body=body,
        analysis_id=analysis_id,
        i18n_key=i18n_key,
        i18n_params=i18n_params,
    )
    db.add(note)
    db.flush()
    return note
