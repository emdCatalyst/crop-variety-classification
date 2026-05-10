from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..models import User
from .config import get_settings
from .db import SessionLocal, get_db
from .security import decode_token


def _read_session_cookie(req_cookie: str | None) -> str | None:
    return req_cookie


def get_current_user(
    db: Session = Depends(get_db),
    session: str | None = Cookie(default=None, alias="agrovision_session"),
) -> User:
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(session)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    user = db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account unavailable")
    # Token-version check — bumped on password change so old JWTs stop working
    # immediately after the user rotates their password.
    if int(payload.get("tv", 0)) != int(user.token_version):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired"
        )
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user


def get_user_for_stream(
    session: str | None = Cookie(default=None, alias="agrovision_session"),
) -> User:
    """Auth dep for SSE endpoints.

    FastAPI keeps generator-style deps (`get_db`) open for the lifetime of the
    response. SSE responses are long-lived, so reusing `get_current_user`
    here would pin a SQLAlchemy connection per open stream and exhaust the
    pool after a handful of clients. This variant opens its own session,
    authenticates, eager-reads the attributes we'll need downstream, and
    closes the session immediately — leaving zero DB connections held while
    the stream idles.
    """
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(session)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    db = SessionLocal()
    try:
        user = db.get(User, int(payload["sub"]))
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account unavailable"
            )
        if int(payload.get("tv", 0)) != int(user.token_version):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired"
            )
        # Touch attributes so they materialize before the session closes.
        _ = user.id, user.role, user.is_active, user.display_name, user.email
        db.expunge(user)
        return user
    finally:
        db.close()


def get_settings_dep():
    return get_settings()
