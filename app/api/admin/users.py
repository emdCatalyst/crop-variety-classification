"""Admin user management — list, update, delete."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.db import get_db
from ...core.deps import require_admin
from ...models import User

router = APIRouter(prefix="/users")


class AdminUserOut(BaseModel):
    id: int
    email: str
    display_name: str
    role: str
    is_active: bool
    language: str
    created_at: datetime

    class Config:
        from_attributes = True


class AdminUserUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    role: str | None = Field(default=None, pattern="^(user|admin)$")
    is_active: bool | None = None


@router.get("", response_model=list[AdminUserOut])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> list[User]:
    return db.query(User).order_by(User.created_at.desc()).all()


@router.patch("/{user_id}", response_model=AdminUserOut)
def update_user(
    user_id: int,
    payload: AdminUserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> User:
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target.id == admin.id:
        if payload.is_active is False:
            raise HTTPException(status_code=400, detail="Refusing to deactivate yourself")
        if payload.role is not None and payload.role != "admin":
            raise HTTPException(status_code=400, detail="Refusing to demote yourself")

    if payload.display_name is not None:
        cleaned = payload.display_name.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="display_name cannot be empty")
        target.display_name = cleaned[:120]

    if payload.role is not None:
        target.role = payload.role

    if payload.is_active is not None:
        target.is_active = payload.is_active

    db.commit()
    db.refresh(target)
    return target


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if target.id == admin.id:
        raise HTTPException(status_code=400, detail="Refusing to delete yourself")
    db.delete(target)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
