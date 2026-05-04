#!/usr/bin/env python3
"""Seed an admin user. Required because admin pages are postponed but messaging
needs an admin recipient.

Usage:
    python scripts/create_admin.py <email> <password> [display_name]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.db import SessionLocal  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models import User  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    display_name = sys.argv[3] if len(sys.argv) > 3 else "Admin"

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            existing.role = "admin"
            existing.is_active = True
            db.commit()
            print(f"Promoted {email} to admin")
            return
        user = User(
            email=email,
            display_name=display_name,
            password_hash=hash_password(password),
            role="admin",
            is_active=True,
        )
        db.add(user)
        db.commit()
        print(f"Created admin: {email}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
