from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings


def _build_engine() -> Engine:
    s = get_settings()
    url = s.database_url
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    if url.startswith("libsql") or url.startswith("sqlite+libsql"):
        if s.turso_auth_token:
            connect_args = {"auth_token": s.turso_auth_token}
    eng = create_engine(url, connect_args=connect_args, future=True)
    # SQLite ships with foreign-key enforcement OFF per connection, so
    # `ondelete=CASCADE` on messages / notifications wouldn't fire — deleting a
    # user would silently orphan those rows. Turn it on for every new
    # connection. Harmless on libsql/Turso (it's already on by default there).
    if url.startswith("sqlite") or url.startswith("libsql"):
        @event.listens_for(eng, "connect")
        def _enable_sqlite_fk(dbapi_connection, _):  # noqa: ANN001
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
            except Exception:
                # Not all DB-API drivers support PRAGMA; ignore if unsupported.
                pass
    return eng


engine: Engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
