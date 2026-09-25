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

-- Media with nothing to transcribe (e.g. no audio track), keyed by Photos UUID
-- or file path, so later runs don't keep retrying them.
CREATE TABLE IF NOT EXISTS skipped (
    key TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    skipped_at TEXT NOT NULL DEFAULT (datetime('now'))
);

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


# Optional per-video metadata (e.g. from Apple Photos). Added by migration.
VIDEO_META_COLUMNS = ("source", "photos_uuid", "taken_at", "location", "albums")


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
        vcols = {r[1] for r in conn.execute("PRAGMA table_info(videos)").fetchall()}
        for col in VIDEO_META_COLUMNS:
            if col not in vcols:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col} TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_videos_photos_uuid ON videos(photos_uuid)"
        )
        if "notion_page_id" not in vcols:
            conn.execute("ALTER TABLE videos ADD COLUMN notion_page_id TEXT")

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
        meta: dict | None = None,
    ) -> int:
        meta = {k: v for k, v in (meta or {}).items() if k in VIDEO_META_COLUMNS}
        cols = ["path", "filename", "duration", "language", "model", *meta]
        vals = [path, filename, duration, language, model, *meta.values()]
        with self._connect() as conn:
            conn.execute("DELETE FROM videos WHERE path = ?", (path,))
            if meta.get("photos_uuid"):
                conn.execute(
                    "DELETE FROM videos WHERE photos_uuid = ?", (meta["photos_uuid"],)
                )
            cur = conn.execute(
                f"INSERT INTO videos ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                vals,
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

    def model_for(self, path: str) -> str | None:
        """Model that transcribed `path`, or None if it has never been done."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model FROM videos WHERE path = ?", (path,)
            ).fetchone()
            return row[0] if row else None

    def is_skipped(self, key: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM skipped WHERE key = ?", (key,)
            ).fetchone() is not None

    def transcribed_photos_uuids(self) -> set[str]:
        """Photos UUIDs that are done: transcribed, or skipped as untranscribable."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT photos_uuid FROM videos WHERE photos_uuid IS NOT NULL "
                "UNION SELECT key FROM skipped"
            ).fetchall()
            return {r[0] for r in rows}

    def mark_skipped(self, key: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skipped (key, reason) VALUES (?, ?)", (key, reason)
            )

    # ----- Notion sync bookkeeping -----
    def set_notion_page(self, video_id: int, page_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE videos SET notion_page_id = ? WHERE id = ?", (page_id, video_id)
            )

    def videos_not_in_notion(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM videos WHERE notion_page_id IS NULL "
                "ORDER BY COALESCE(taken_at, transcribed_at)"
            ).fetchall()

    def segments_for(self, video_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT start, end, speaker, text FROM segments "
                "WHERE video_id = ? ORDER BY idx", (video_id,)
            ).fetchall()
            return [dict(r) for r in rows]
