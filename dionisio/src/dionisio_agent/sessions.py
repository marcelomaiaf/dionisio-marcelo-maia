from __future__ import annotations

from agents import SQLiteSession
from agents.memory.sqlite_session import SessionSettings


def create_limited_sqlite_session(
    session_id: str,
    *,
    db_path: str,
    history_limit: int | None,
) -> SQLiteSession:
    limit = None if history_limit is None or history_limit <= 0 else history_limit
    return SQLiteSession(
        session_id,
        db_path=db_path,
        session_settings=SessionSettings(limit=limit),
    )
