"""Send transcripts to the Notion "Media Logs" database.

Mirrors the Daily Field Logs setup used by the voice-memo pipeline: same
Category / Tags / Status properties, a "Source Recording" table, and the
transcript under a "Notes — Whisper" heading.

Pages are matched by the "Media ID" property (Apple Photos UUID, or the file
path for dropped files). Re-transcribing adds a new section instead of
overwriting — same additive rule as the voice-memo pipeline.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

DEFAULT_DATABASE_ID = "eec4d180-8681-4a13-9dad-815b8b6b2286"  # Database - Media Logs
NOTION_VERSION = "2022-06-28"
PIPELINE_ENV = Path.home() / "voice-memo-pipeline" / ".env"

_TEXT_LIMIT = 2000   # Notion rich-text limit per block
_BLOCK_BATCH = 100   # Notion append limit per call


def find_token(explicit: str | None = None) -> str | None:
    """Token from the app settings, the environment, or the voice-memo
    pipeline's .env (same integration, so no second setup on this Mac)."""
    if explicit:
        return explicit
    if os.environ.get("NOTION_API_KEY"):
        return os.environ["NOTION_API_KEY"]
    if PIPELINE_ENV.exists():
        from dotenv import dotenv_values
        return dotenv_values(PIPELINE_ENV).get("NOTION_API_KEY") or None
    return None


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _rt(text: str) -> list[dict]:
    return [{"type": "text", "text": {"content": text[:_TEXT_LIMIT]}}]


def _para(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}


def _heading(text: str) -> dict:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": _rt(text)}}


def _table(header: list[str], row: list[str]) -> dict:
    def tr(cells):
        return {"type": "table_row", "table_row": {"cells": [_rt(c or "—") for c in cells]}}
    return {
        "object": "block", "type": "table",
        "table": {
            "table_width": len(header), "has_column_header": True,
            "has_row_header": False, "children": [tr(header), tr(row)],
        },
    }


def transcript_paragraphs(segments: list[dict]) -> list[dict]:
    """One line per segment ("[00:01:23] SPEAKER_00: text"), packed into
    paragraphs under Notion's 2000-character limit."""
    lines = []
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        spk = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"[{_fmt_ts(s['start'])}] {spk}{text}")

    blocks, buf = [], ""
    for line in lines:
        line = line[:_TEXT_LIMIT]
        if buf and len(buf) + 1 + len(line) > _TEXT_LIMIT:
            blocks.append(_para(buf))
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        blocks.append(_para(buf))
    return blocks or [_para("(no speech detected)")]


class NotionSync:
    def __init__(self, token: str, database_id: str = DEFAULT_DATABASE_ID):
        from notion_client import Client
        self.client = Client(auth=token, notion_version=NOTION_VERSION)
        self.database_id = database_id

    def check(self) -> None:
        """Raise a readable error if the integration can't see the database."""
        from notion_client.errors import APIResponseError
        try:
            self.client.databases.retrieve(database_id=self.database_id)
        except APIResponseError as exc:
            if exc.status in (403, 404):
                raise RuntimeError(
                    "Notion can't see the Media Logs database. In Notion open "
                    "'Database - Media Logs' > ••• > Connections and add the same "
                    "integration the voice-memo pipeline uses."
                ) from exc
            raise

    def _find_page(self, media_id: str) -> str | None:
        # Raw request: notion-client 3.x dropped databases.query().
        res = self.client.request(
            path=f"databases/{self.database_id}/query",
            method="POST",
            body={
                "filter": {"property": "Media ID", "rich_text": {"equals": media_id}},
                "page_size": 1,
            },
        )
        return res["results"][0]["id"] if res["results"] else None

    def set_content_hash(self, page_id: str, sha256: str) -> None:
        """Stamp the content hash on an existing page (backfill_hashes.py)."""
        self.client.pages.update(
            page_id=page_id, properties={"Content Hash": {"rich_text": _rt(sha256)}}
        )

    def push(
        self,
        *,
        media_id: str,
        title: str,
        segments: list[dict],
        duration: float,
        language: str | None,
        model: str | None,
        meta: dict | None = None,
        file_path: str | None = None,
        add_to_existing: bool = True,
        content_hash: str | None = None,
    ) -> str:
        """Create (or add to) the page for one transcript. Returns page id.

        add_to_existing=False: if a page already exists, just return its id
        (used by the backfill so an interrupted run never duplicates content).
        """
        meta = meta or {}
        from_photos = meta.get("source") == "photos"
        filename = meta.get("original_filename") or (Path(file_path).name if file_path else title)
        taken = meta.get("taken_at")
        if not taken and file_path and os.path.exists(file_path):
            taken = datetime.fromtimestamp(os.path.getmtime(file_path)).astimezone().isoformat()
        day = (taken or date.today().isoformat())[:10]
        speakers = sorted({s["speaker"] for s in segments if s.get("speaker")})
        empty = not any(s["text"].strip() for s in segments)
        length = _fmt_ts(duration)[3:] if duration < 3600 else _fmt_ts(duration)

        props = {
            "Day": {"title": _rt(title)},
            "Date": {"date": {"start": day}},
            "Media ID": {"rich_text": _rt(media_id)},
            "Source": {"select": {"name": "Apple Photos" if from_photos else "File"}},
            "File": {"rich_text": _rt(filename)},
            "File Path": {"rich_text": _rt(file_path or "")},
            "Length": {"rich_text": _rt(length)},
            "Language": {"rich_text": _rt(language or "")},
            "Model": {"rich_text": _rt(model or "")},
            "Location": {"rich_text": _rt(meta.get("location") or "")},
            "Albums": {"rich_text": _rt(meta.get("albums") or "")},
            "Speakers": {"number": len(speakers)},
            "Has Whisper": {"checkbox": True},
            "Has Speakers": {"checkbox": bool(speakers)},
            "Empty": {"checkbox": empty},
        }
        if content_hash:
            props["Content Hash"] = {"rich_text": _rt(content_hash)}

        notes_title = "Notes — Whisper"
        page_id = self._find_page(media_id)
        if page_id and not add_to_existing:
            return page_id
        if page_id:
            self.client.pages.update(page_id=page_id, properties=props)
            notes_title += f" (re-run {date.today().isoformat()}, {model})"
            blocks = []
        else:
            props.update({
                "Status": {"select": {"name": "Draft"}},
                "Category": {"select": {"name": "Unsorted"}},
                "Tags": {"multi_select": [{"name": "Video"}]},
            })
            page_id = self.client.pages.create(
                parent={"database_id": self.database_id},
                icon={"type": "emoji", "emoji": "🎬"},
                properties=props,
            )["id"]
            blocks = [
                _heading("Source Recording"),
                _table(
                    ["Date", "File", "Length", "Source", "Location"],
                    [day, filename, length,
                     "Apple Photos" if from_photos else "File",
                     meta.get("location") or ""],
                ),
            ]

        blocks += [_heading(notes_title), *transcript_paragraphs(segments)]
        for i in range(0, len(blocks), _BLOCK_BATCH):
            self.client.blocks.children.append(
                block_id=page_id, children=blocks[i:i + _BLOCK_BATCH]
            )
        return page_id
