#!/usr/bin/env python3
"""Re-transcribe multi-voice recordings with speaker diarization.

Headless companion to the PyQt app. Targets only the recordings that plausibly
contain more than one speaker -- running diarization over solo selfie journals
costs full WhisperX + pyannote time and returns SPEAKER_00 for every segment.

Ordering is shortest-first and deliberate: the small files establish a real
throughput number within the first few minutes, so you can extrapolate the ETA
for the multi-hour files and abort before committing days of CPU.

Run it through ./rerun.sh, not bare python3 -- it needs the project venv and
the ffmpeg@7 libs, exactly like run.sh.

Usage:
    ./rerun.sh --dry-run          # preflight, touches nothing
    ./rerun.sh                    # run
    ./rerun.sh --limit 3          # first 3 (shortest) only
    ./rerun.sh --model base       # faster, lower quality

Safe to interrupt (Ctrl-C) and re-run: completed files are detected from the DB
and skipped. The database is backed up before the first write.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "MediaTranscriber"
DEFAULT_DB = APP_SUPPORT / "transcripts.sqlite"
DEFAULT_SETTINGS = APP_SUPPORT / "settings.json"

# Recordings judged multi-voice: phone calls, crew conversations, and the long
# "Recording NN" sessions. Matched by filename so the list survives re-indexing
# and re-imports -- video row IDs are not stable across a DB rebuild.
MULTI_VOICE_PATTERNS = (
    "Crew Camp",
    "Highland Dr",
    "OJ OATS CALL",
    "NYC ",
    "Recording 1",
    "Recording 2",
)


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def fmt_dur(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def load_settings(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def is_multi_voice(filename: str) -> bool:
    return any(p.lower() in filename.lower() for p in MULTI_VOICE_PATTERNS)


def gather_targets(db_path: Path, redo_all: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (targets, duplicates, done) -- shortest-first.

    Duplicates are detected by rounded duration: the corpus contains the same
    recording filed under several names (Highland Dr 31 == OJ OATS CALL SPRING
    '25, five rows, one conversation). Only the canonical copy is transcribed.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT v.id, v.path, v.filename, v.duration, v.model,
               SUM(CASE WHEN s.speaker IS NOT NULL THEN 1 ELSE 0 END) AS spk
        FROM videos v LEFT JOIN segments s ON s.video_id = v.id
        GROUP BY v.id
        ORDER BY v.duration
        """
    ).fetchall()
    conn.close()

    targets: list[dict] = []
    duplicates: list[dict] = []
    done: list[dict] = []
    seen_duration: dict[int, str] = {}

    for r in rows:
        if not is_multi_voice(r["filename"]):
            continue
        item = {
            "id": r["id"],
            "path": Path(r["path"]),
            "filename": r["filename"],
            "duration": r["duration"] or 0.0,
            "model": r["model"],
            "speaker_rows": r["spk"] or 0,
        }

        if item["speaker_rows"] > 0 and not redo_all:
            done.append(item)
            continue

        # Sub-minute clips are never meaningful duplicates of each other.
        key = int(round(item["duration"]))
        if item["duration"] > 60 and key in seen_duration:
            item["duplicate_of"] = seen_duration[key]
            duplicates.append(item)
            continue
        if item["duration"] > 60:
            seen_duration[key] = item["filename"]
            # Tolerate 1-2s container differences (m4a vs mp3 of one recording).
            for delta in (-2, -1, 1, 2):
                seen_duration.setdefault(key + delta, item["filename"])

        targets.append(item)

    return targets, duplicates, done


def preflight(targets: list[dict]) -> tuple[list[dict], list[dict]]:
    present, missing = [], []
    for t in targets:
        (present if t["path"].exists() else missing).append(t)
    return present, missing


def backup_db(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = db_path.with_name(f"{db_path.stem}.backup-{stamp}.sqlite")
    shutil.copy2(db_path, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    ap.add_argument("--model", default=None,
                    help="whisper model (default: from settings.json, else medium)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    ap.add_argument("--redo-all", action="store_true",
                    help="re-run even files that already have speaker labels")
    ap.add_argument("--skip-missing", action="store_true",
                    help="proceed even if some source files are unreachable")
    args = ap.parse_args()

    if not args.db.exists():
        log(f"ERROR: no database at {args.db}")
        return 1

    settings = load_settings(args.settings)
    model = args.model or settings.get("model") or "medium"
    hf_token = settings.get("hf_token")

    if not hf_token:
        log(f"ERROR: no hf_token in {args.settings} -- diarization cannot run.")
        return 1

    targets, duplicates, done = gather_targets(args.db, args.redo_all)
    present, missing = preflight(targets)

    print()
    log(f"database : {args.db}")
    log(f"model    : {model}   (diarization ON)")
    print()

    if done:
        log(f"already diarized, skipping ({len(done)}):")
        for d in done:
            print(f"       {d['filename'][:52]:54} {d['speaker_rows']} speaker rows")
    if duplicates:
        log(f"duplicate audio, skipping ({len(duplicates)}):")
        for d in duplicates:
            print(f"       {d['filename'][:52]:54} == {d['duplicate_of'][:34]}")
    if missing:
        log(f"UNREACHABLE ({len(missing)}) -- external volume or Drive not mounted?:")
        for m in missing:
            print(f"       {m['filename'][:52]:54} {m['path'].parent}")

    # A dry run shows the whole plan even if volumes are unmounted -- that is
    # the point of it. A real run only touches what it can actually read.
    queue = targets if args.dry_run else present
    if args.limit:
        queue = queue[: args.limit]

    if not queue:
        print()
        log("Nothing to do.")
        return 0

    total_audio = sum(t["duration"] for t in queue)
    print()
    log(f"queued ({len(queue)}), shortest first -- {fmt_dur(total_audio)} of audio:")
    for t in queue:
        notes = []
        if t["duration"] > 7200:
            notes.append("long, pyannote memory risk")
        if not t["path"].exists():
            notes.append("UNREACHABLE")
        flag = ("  <-- " + "; ".join(notes)) if notes else ""
        print(f"       {fmt_dur(t['duration']):>7}  {t['filename'][:48]:50}"
              f" was:{t['model'] or '?':<7}{flag}")

    if args.dry_run:
        print()
        log("Dry run. Nothing written.")
        return 0

    # Unreachable sources are only fatal for a real run -- a half-complete pass
    # is worse than not starting, because the long files are at the end.
    if missing and not args.skip_missing:
        print()
        log("Refusing to start with unreachable files. Mount the volumes, or pass")
        log("--skip-missing to process only what is available.")
        return 1

    # Import before backing up: a failed import should not litter the disk with
    # a backup of a database nothing was going to touch.
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from app.db import Database
        from app.transcriber import Transcriber
    except ModuleNotFoundError as exc:
        print()
        log(f"ERROR: missing dependency '{exc.name}'.")
        log("This needs the project venv, not system python. Use ./rerun.sh,")
        log("which sets the same interpreter and FFmpeg paths as run.sh:")
        log("    ./rerun.sh " + " ".join(sys.argv[1:]))
        return 1

    print()
    backup = backup_db(args.db)
    log(f"database backed up -> {backup.name}")

    db = Database(args.db)
    transcriber = Transcriber(model_name=model, diarize=True, hf_token=hf_token)

    ok = failed = 0
    audio_done = 0.0
    run_start = time.time()

    for i, t in enumerate(queue, 1):
        print()
        log(f"[{i}/{len(queue)}] {t['filename']}  ({fmt_dur(t['duration'])})")
        started = time.time()
        try:
            result = transcriber.transcribe(
                t["path"],
                t["path"].parent / "transcriptions",
                progress_cb=lambda m: log(f"        {m}"),
            )
        except KeyboardInterrupt:
            print()
            log("Interrupted. Completed files are saved; re-run to resume.")
            return 130
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log(f"        FAILED: {type(exc).__name__}: {exc}")
            continue

        video_id = db.upsert_video(
            path=str(result.video_path),
            filename=result.video_path.name,
            duration=result.duration,
            language=result.language,
            model=result.model_name,
        )
        db.insert_segments(video_id, result.segments)

        speakers = {s.get("speaker") for s in result.segments if s.get("speaker")}
        elapsed = time.time() - started
        ratio = elapsed / t["duration"] if t["duration"] else 0
        ok += 1
        audio_done += t["duration"]

        log(f"        done in {fmt_dur(elapsed)} ({ratio:.2f}x realtime) -- "
            f"{len(result.segments)} segments, {len(speakers)} speaker(s): "
            f"{', '.join(sorted(speakers)) or 'none detected'}")

        remaining = sum(x["duration"] for x in queue[i:])
        if remaining and audio_done:
            rate = (time.time() - run_start) / audio_done
            eta = timedelta(seconds=int(remaining * rate))
            log(f"        {fmt_dur(remaining)} of audio left -- projected {eta} "
                f"at current rate")

    print()
    log(f"Finished: {ok} transcribed, {failed} failed.")
    if failed:
        log(f"Re-run to retry the failures; successes will be skipped.")
    log(f"Backup retained at {backup}")
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        log("Interrupted.")
        sys.exit(130)
