"""On-disk artifact cleanup for analysis and user deletions.

Database rows are handled by SQLAlchemy ORM cascade plus SQLite/libsql
``ondelete=CASCADE`` (enabled in ``app/core/db.py``). What the DB cascade
does *not* touch is the filesystem: uploaded GeoTIFFs, rendered map / confidence
PNGs, saved mosaic GeoTIFFs, and message attachments.

These helpers collect those paths from a still-attached row and unlink them
best-effort. Callers should invoke them either before ``db.delete()`` (so the
relationships are still loadable) or after committing — failures here never
roll the DB back.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models import Analysis, Message, User


def _unlink(path: str | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("failed to unlink {}: {}", path, exc)


def _rmtree(path: str | None) -> None:
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)


def purge_analysis_artifacts(a: Analysis) -> None:
    """Remove every on-disk file that belongs to this analysis."""
    _rmtree(a.upload_dir)
    if a.result is not None:
        _unlink(a.result.map_png_path)
        _unlink(a.result.confidence_png_path)
        _unlink(a.result.geotiff_path)


def purge_user_artifacts(user: User, db: Session) -> None:
    """Remove every on-disk file owned by the user before the row is deleted.

    Covers all of the user's analyses (uploads + maps + saved geotiffs) and
    every message attachment they sent or received. Safe to call before
    ``db.delete(user)`` — only touches the filesystem.
    """
    for a in list(user.analyses):
        purge_analysis_artifacts(a)

    msgs_with_files = (
        db.query(Message)
        .filter(
            or_(Message.sender_id == user.id, Message.recipient_id == user.id),
            Message.attachment_path.isnot(None),
        )
        .all()
    )
    for m in msgs_with_files:
        _unlink(m.attachment_path)
