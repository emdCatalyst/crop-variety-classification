"""One-time-passcode helpers for email verification and password reset.

Codes are 6-digit numeric, hashed with SHA-256 keyed by ``jwt_secret`` (the
short lifetime, narrow attempt cap, and rate limiting on the auth endpoints
make a heavier KDF unnecessary). Only one OTP slot exists per user — the
``otp_purpose`` column distinguishes verification vs reset.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models import User

VERIFY = "verify"
RESET = "reset"
_MAX_ATTEMPTS = 5


def generate_code() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(6))


def _hash(code: str) -> str:
    secret = get_settings().jwt_secret
    return hashlib.sha256((code + secret).encode("utf-8")).hexdigest()


def set_otp(user: User, purpose: str, db: Session) -> str:
    """Generate a fresh code for the user, persist its hash, return the plaintext.

    The caller is responsible for emailing the returned code.
    """
    code = generate_code()
    ttl = get_settings().email_code_ttl_minutes
    user.otp_hash = _hash(code)
    user.otp_purpose = purpose
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl)
    user.otp_attempts = 0
    db.commit()
    db.refresh(user)
    return code


def consume_otp(user: User, code: str, purpose: str, db: Session) -> bool:
    """Validate a submitted code. On success, clear the OTP slot and commit.

    On failure, increment ``otp_attempts`` and commit so subsequent attempts
    move toward the cap. Returns False for any failure (wrong purpose,
    expired, too many attempts, mismatch, no OTP set).
    """
    if not user.otp_hash or not user.otp_expires_at or user.otp_purpose != purpose:
        return False
    if user.otp_attempts >= _MAX_ATTEMPTS:
        return False
    expires = user.otp_expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return False
    if not hmac.compare_digest(_hash(code), user.otp_hash):
        user.otp_attempts += 1
        db.commit()
        return False
    user.otp_hash = None
    user.otp_purpose = None
    user.otp_expires_at = None
    user.otp_attempts = 0
    db.commit()
    return True
