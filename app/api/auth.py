from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.db import get_db
from ..core.deps import get_current_user
from ..core.rate_limit import limiter
from ..core.security import create_access_token, hash_password, verify_password
from ..models import User
from ..schemas.auth import (
    ForgotPasswordIn,
    LoginIn,
    ResendVerificationIn,
    ResetPasswordIn,
    SignupIn,
    SignupOut,
    UserOut,
    VerifyEmailIn,
)
from ..services.emailer import send_password_reset_email, send_verification_email
from ..services.otp import RESET, VERIFY, consume_otp, set_otp

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(
        key=s.cookie_name,
        value=token,
        httponly=True,
        secure=s.cookie_secure,
        samesite=s.cookie_samesite,
        max_age=s.jwt_ttl_minutes * 60,
        path="/",
    )


def _issue_session(response: Response, user: User) -> None:
    _set_cookie(response, create_access_token(user.id, {"tv": user.token_version}))


@router.post("/signup", response_model=SignupOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
def signup(
    request: Request,
    payload: SignupIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> SignupOut:
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = User(
        email=payload.email,
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
        language=payload.language,
        role="user",
        email_verified=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    code = set_otp(user, VERIFY, db)
    background.add_task(send_verification_email, user, code, user.language)
    return SignupOut(email=user.email)


@router.post("/login", response_model=UserOut)
@limiter.limit("20/minute")
def login(
    request: Request,
    payload: LoginIn,
    response: Response,
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account deactivated")
    if not user.email_verified:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="email_not_verified")
    _issue_session(response, user)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout() -> Response:
    s = get_settings()
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    # Mirror the attributes used in _set_cookie so the browser actually matches
    # and clears the cookie (path, secure, samesite, httponly all factor in).
    resp.delete_cookie(
        key=s.cookie_name,
        path="/",
        secure=s.cookie_secure,
        samesite=s.cookie_samesite,
        httponly=True,
    )
    return resp


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post("/verify-email", response_model=UserOut)
@limiter.limit("10/hour")
def verify_email(
    request: Request,
    payload: VerifyEmailIn,
    response: Response,
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_or_expired_code")
    if user.email_verified:
        # Already verified — issue a session and return so the user can proceed.
        _issue_session(response, user)
        return user
    if not consume_otp(user, payload.code, VERIFY, db):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_or_expired_code")
    user.email_verified = True
    db.commit()
    db.refresh(user)
    _issue_session(response, user)
    return user


@router.post("/resend-verification", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("3/hour")
def resend_verification(
    request: Request,
    payload: ResendVerificationIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Response:
    user = db.query(User).filter(User.email == payload.email).first()
    if user and user.is_active and not user.email_verified:
        code = set_otp(user, VERIFY, db)
        background.add_task(send_verification_email, user, code, user.language)
    # Always 204 — don't disclose whether the address is registered.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/forgot-password", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("5/hour")
def forgot_password(
    request: Request,
    payload: ForgotPasswordIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Response:
    user = db.query(User).filter(User.email == payload.email).first()
    if user and user.is_active:
        code = set_otp(user, RESET, db)
        background.add_task(send_password_reset_email, user, code, user.language)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/reset-password", response_model=UserOut)
@limiter.limit("10/hour")
def reset_password(
    request: Request,
    payload: ResetPasswordIn,
    response: Response,
    db: Session = Depends(get_db),
) -> User:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_or_expired_code")
    if not consume_otp(user, payload.code, RESET, db):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_or_expired_code")
    user.password_hash = hash_password(payload.new_password)
    # Bump token_version so any other live sessions are invalidated.
    user.token_version = (user.token_version or 1) + 1
    # If the account was unverified, reaching the reset email proves ownership.
    user.email_verified = True
    db.commit()
    db.refresh(user)
    _issue_session(response, user)
    return user
