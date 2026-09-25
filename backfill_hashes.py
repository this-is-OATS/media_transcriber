#!/usr/bin/env python3
"""Stamp a SHA-256 content hash on every existing transcript that lacks one.

Media Transcription Plan, Phase 0: rows written before 2026-09-24 locate their
video by path only, and the Phase 2b NAS move would orphan them. This walks
the sqlite for file-backed rows with no ``sha256``, hashes the file if it is
still where the row says, and writes the hash to sqlite and to the row's
Media Logs page ("Content Hash"). Photos-only rows (``photos://<uuid>``) have
no local file; their UUID is already their stable id and they are left alone.

    .venv/bin/python backfill_hashes.py            # do it
    .venv/bin/python backfill_hashes.py --dry-run  # report only

Paths under ~/Library/CloudStorage (Google Drive streaming) are SKIPPED by
default: reading one pulls the whole video down onto a boot drive with ~16 GB
free. Pass --include-cloud once there is room, or after the files are moved
to the NAS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.db import Database
from app.notion_sync import NotionSync, find_token
from batch import DB_PATH, log, sha256_of

CLOUD = str(Path.home() / "Library" / "CloudStorage")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-cloud", action="store_true")
    ap.add_argument("--no-notion", action="store_true")
    args = ap.parse_args()

    db = Database(DB_PATH)
    rows = db.videos_without_sha256()
    log(f"{len(rows)} file-backed rows without a content hash")
    notion = None
    if not args.no_notion and not args.dry_run:
        token = find_token()
        if not token:
            log("ERROR: no Notion token; use --no-notion")
            return 2
        notion = NotionSync(token)

    done = missing = cloud = failed = 0
    for r in rows:
        path = Path(r["path"])
        if str(path).startswith(CLOUD) and not args.include_cloud:
            log(f"  cloud, skipped: {path}")
            cloud += 1
            continue
        if not path.exists():
            log(f"  MISSING: {path}")
            missing += 1
            continue
        if args.dry_run:
            log(f"  would hash: {path}")
            done += 1
            continue
        try:
            digest = sha256_of(path)
            db.set_sha256(r["id"], digest)
            if notion is not None and r["notion_page_id"]:
                notion.set_content_hash(r["notion_page_id"], digest)
            log(f"  {digest[:12]}…  {path}")
            done += 1
        except Exception as exc:  # noqa: BLE001
            log(f"  FAILED {path}: {exc}")
            failed += 1
    log(f"SUMMARY hashed={done} missing={missing} cloud_skipped={cloud} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
