from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

# Database location:
# backend/runtime/veriframe.db
BACKEND_DIR = Path(__file__).resolve().parents[2]

DB_PATH = (
    Path(
        os.getenv(
            "VERIFRAME_DB_PATH",
            str(BACKEND_DIR / "runtime" / "veriframe.db"),
        )
    )
    .expanduser()
    .resolve()
)

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = RLock()


def _connect() -> sqlite3.Connection:
    """Create a connection to the VeriFrame SQLite database."""
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _initialize_database() -> None:
    """Create the database tables if they do not already exist."""
    with _lock:
        with _connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    events_json TEXT NOT NULL DEFAULT '[]'
                )
                """)

            connection.commit()


def _serialize(data: Any) -> str:
    """Convert Python data into JSON."""
    return json.dumps(data, default=str)


def _deserialize(data: str) -> Any:
    """Convert JSON back into Python data."""
    return json.loads(data)


def create(analysis_id: str, record: dict) -> None:
    """Create and permanently save a new analysis."""
    with _lock:
        with _connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analyses
                (analysis_id, record_json, events_json)
                VALUES (?, ?, ?)
                """,
                (
                    analysis_id,
                    _serialize(record),
                    "[]",
                ),
            )
            connection.commit()


def get(analysis_id: str):
    """Retrieve an analysis by its ID."""
    with _lock:
        with _connect() as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM analyses
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()

            if row is None:
                return None

            return _deserialize(row["record_json"])


def update(analysis_id: str, **fields) -> None:
    """Update analysis fields and save the changes."""
    with _lock:
        with _connect() as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM analyses
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()

            if row is None:
                return

            record = _deserialize(row["record_json"])
            record.update(fields)

            connection.execute(
                """
                UPDATE analyses
                SET record_json = ?
                WHERE analysis_id = ?
                """,
                (
                    _serialize(record),
                    analysis_id,
                ),
            )

            connection.commit()


def add_event(
    analysis_id: str,
    stage: str,
    progress: int,
    message: str,
) -> None:
    """Update analysis progress and persist its event."""
    event = {
        "stage": stage,
        "progress": progress,
        "message": message,
    }

    with _lock:
        with _connect() as connection:
            row = connection.execute(
                """
                SELECT record_json, events_json
                FROM analyses
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()

            if row is None:
                return

            record = _deserialize(row["record_json"])
            events = _deserialize(row["events_json"])

            record["stage"] = stage
            record["progress"] = progress

            events.append(event)

            connection.execute(
                """
                UPDATE analyses
                SET record_json = ?, events_json = ?
                WHERE analysis_id = ?
                """,
                (
                    _serialize(record),
                    _serialize(events),
                    analysis_id,
                ),
            )

            connection.commit()


def events_after(analysis_id: str, index: int):
    """Return analysis events after the supplied index."""
    with _lock:
        with _connect() as connection:
            row = connection.execute(
                """
                SELECT events_json
                FROM analyses
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()

            if row is None:
                return []

            events = _deserialize(row["events_json"])
            return events[index:]


def delete(analysis_id: str):
    """Permanently delete an analysis and its associated events."""
    with _lock:
        with _connect() as connection:
            connection.execute(
                """
                DELETE FROM analyses
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            )
            connection.commit()


def all_records():
    """Return all saved analysis records."""
    with _lock:
        with _connect() as connection:
            rows = connection.execute("""
                SELECT record_json
                FROM analyses
                ORDER BY rowid
                """).fetchall()

            return [_deserialize(row["record_json"]) for row in rows]


# Initialize SQLite when the service is imported.
_initialize_database()
