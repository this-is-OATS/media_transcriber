"""SQLite database for transcripts and full-text search."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    duration        REAL,
    language        TEXT,
    model           TEXT,
    transcribed_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    start       REAL    NOT NULL,
    end         REAL    NOT NULL,
    speaker     TEXT,
    text        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(segments)").fetchall()}
        if "speaker" not in cols:
            conn.execute("ALTER TABLE segments ADD COLUMN speaker TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_video(
        self,
        path: str,
        filename: str,
        duration: float | None,
        language: str | None,
        model: str | None,
    ) -> int:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM videos WHERE path = ?",
                (path,),
            )
            cur = conn.execute(
                "INSERT INTO videos (path, filename, duration, language, model) "
                "VALUES (?, ?, ?, ?, ?)",
                (path, filename, duration, language, model),
            )
            return cur.lastrowid

    def insert_segments(
        self,
        video_id: int,
        segments: Iterable[dict],
    ) -> None:
        rows = [
            (video_id, i, s["start"], s["end"], s.get("speaker"), s["text"].strip())
            for i, s in enumerate(segments)
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO segments (video_id, idx, start, end, speaker, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def search(self, query: str, limit: int = 100) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT v.path, v.filename, s.start, s.end, s.speaker, s.text,
                       snippet(segments_fts, 0, '[', ']', '...', 16) AS snippet
                FROM segments_fts
                JOIN segments s ON s.id = segments_fts.rowid
                JOIN videos   v ON v.id = s.video_id
                WHERE segments_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

    def list_videos(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM videos ORDER BY transcribed_at DESC"
            ).fetchall()

    def has_video(self, path: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM videos WHERE path = ?", (path,)
            ).fetchone()
            return row is not None
