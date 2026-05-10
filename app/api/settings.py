"""User settings: display name, language preference, password change."""
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..core.security import create_access_token, hash_password, verify_password
from ..models import User
from ..schemas.auth import UserOut

router = APIRouter(prefix="/settings", tags=["settings"])


class ProfileUpdateIn(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    language: str | None = Field(default=None, pattern="^(en|fr|ar)$")


class PasswordChangeIn(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


@router.patch("/profile", response_model=UserOut)
def update_profile(
    payload: ProfileUpdateIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> User:
    if payload.display_name is not None:
        user.display_name = payload.display_name
    if payload.language is not None:
        user.language = payload.language
    db.commit()
    db.refresh(user)
    return user


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: PasswordChangeIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    user.password_hash = hash_password(payload.new_password)
    # Invalidate every JWT issued under the previous password — including any
    # other device this user might be signed in on.
    user.token_version = (user.token_version or 1) + 1
    db.commit()
    db.refresh(user)
    # Re-issue the cookie on this device so the caller stays signed in.
    s = get_settings()
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.set_cookie(
        key=s.cookie_name,
        value=create_access_token(user.id, {"tv": user.token_version}),
        httponly=True,
        secure=s.cookie_secure,
        samesite=s.cookie_samesite,
        max_age=s.jwt_ttl_minutes * 60,
        path="/",
    )
    return resp
