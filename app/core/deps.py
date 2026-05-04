from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..models import User
from .config import get_settings
from .db import get_db
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
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user


def get_settings_dep():
    return get_settings()
