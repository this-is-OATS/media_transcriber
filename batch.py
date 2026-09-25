#!/usr/bin/env python3
"""Headless batch runner — the Media Transcriber app's engine with no window.

Phase 0 of the Media Transcription Plan (Notion thread "Digital filing & sync —
media library"). Same sqlite, same markdown layout, same Notion writer as the
app, so a file done here is indistinguishable from one dropped on the window.
What differs is deliberate and pinned:

  * one model for the whole backfill (``--model``, default large-v3-turbo),
    stamped on every Media Logs row;
  * faster-whisper with voice-activity detection ON and NO conditioning on the
    previous segment's text — the anti-repetition pair from the plan's Phase 3;
  * the audio track is extracted once with ffmpeg (16 kHz mono) and that is
    what gets transcribed, so a 4 GB video is read once, not shuttled around;
  * a whole-file SHA-256 is stamped on every row (sqlite ``sha256`` and the
    Media Logs ``Content Hash`` property) from the first one, so the NAS
    sort-and-move (plan Phase 2b) can re-link transcripts by content.

    .venv/bin/python batch.py ROOT [ROOT ...]     roots run in the order given
        --model large-v3-turbo    pinned model name (faster-whisper)
        --stop-after HH:MM        start no new file after this local time
        --limit N                 at most N files this run
        --dry-run                 print the queue, transcribe nothing
        --no-notion               local sqlite + markdown only
        --pending                 print the queue count and exit

Idempotent and resumable: a file already in the sqlite with this model is
skipped, one done with another model is re-run (Notion gets an additional
"Notes — Whisper (re-run …)" section — the app's additive rule, nothing is
overwritten), and one marked "no audio track" is never retried. Kill it any
time; the write happens only after a file finishes.

Prints a single machine-readable SUMMARY line at the end for the wrapper
(run-media-transcribe.sh in voice-memo-pipeline) to push to the phone.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

from app.db import Database                          # noqa: E402
from app.notion_sync import NotionSync, find_token   # noqa: E402
from app.transcriber import Transcriber, find_media_files  # noqa: E402

APP_DATA = Path.home() / "Library" / "Application Support" / "MediaTranscriber"
DB_PATH = APP_DATA / "transcripts.sqlite"
AUDIO_CACHE = Path.home() / "Library" / "Caches" / "MediaTranscriber" / "audio"
DEFAULT_MODEL = "large-v3-turbo"

_stop = False


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def _on_term(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log("SIGTERM — finishing nothing more, exiting after this file's cleanup")
    raise SystemExit(143)


def ffprobe(path: Path) -> tuple[float | None, bool]:
    """(duration seconds, has audio stream). Any probe failure -> (None, True):
    let the transcriber report the real error rather than silently skipping."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, True
    if out.returncode != 0:
        return None, True
    duration, has_audio = None, False
    for line in out.stdout.splitlines():
        line = line.strip().rstrip(",")
        if line == "audio":
            has_audio = True
        elif line and line[0].isdigit():
            try:
                duration = float(line)
            except ValueError:
                pass
    return duration, has_audio


def sha256_of(path: Path) -> str:
    """Whole-file SHA-256 — the plan's stable identity for a loose file. A
    10 GB video over USB is ~30 s; the transcript takes minutes, so it is
    cheap next to the work it protects (the NAS move re-links by this)."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def extract_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", "-f", "wav", str(dst)],
        check=True, timeout=3600,
    )


def deadline_from(hhmm: str | None) -> datetime | None:
    if not hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    now = datetime.now()
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= now:
        d += timedelta(days=1)
    return d


def fmt_h(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def collect(roots: list[Path]) -> list[Path]:
    """Roots in the order given (smallest bucket first is the caller's job);
    inside a root, smallest file first — the same rule one level down."""
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            log(f"WARNING: root missing, skipped: {root}")
            continue
        batch = [f for f in find_media_files(root) if f not in seen]
        batch.sort(key=lambda f: f.stat().st_size)
        files.extend(batch)
        seen.update(batch)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stop-after", metavar="HH:MM")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-notion", action="store_true")
    ap.add_argument("--pending", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    db = Database(DB_PATH)
    files = collect(args.roots)

    queue, already, noaudio_known = [], 0, 0
    for f in files:
        key = str(f)
        if db.is_skipped(key):
            noaudio_known += 1
        elif db.model_for(key) == args.model:
            already += 1
        else:
            queue.append(f)
    log(f"model={args.model}  found={len(files)}  done_with_model={already}  "
        f"known_no_audio={noaudio_known}  pending={len(queue)}")
    if args.pending:
        print(f"PENDING {len(queue)}")
        return 0
    if args.dry_run:
        for f in queue:
            prev = db.model_for(str(f))
            log(f"  would run: {f}  ({f.stat().st_size / 1e9:.2f} GB"
                f"{', re-run: was ' + prev if prev else ''})")
        return 0

    notion = None
    if not args.no_notion:
        token = find_token()
        if not token:
            log("ERROR: no Notion token (NOTION_API_KEY / voice-memo-pipeline/.env); "
                "use --no-notion to run local-only")
            return 2
        notion = NotionSync(token)
        notion.check()

    deadline = deadline_from(args.stop_after)
    if deadline:
        log(f"will start no new file after {deadline:%Y-%m-%d %H:%M}")
    transcriber = Transcriber(model_name=args.model, engine="faster-whisper",
                              cpu_threads=args.cpu_threads)

    done = failed = skipped = 0
    audio_s = 0.0
    t_run = time.time()
    for i, src in enumerate(queue, 1):
        if args.limit and done + failed + skipped >= args.limit:
            log(f"limit {args.limit} reached")
            break
        if deadline and datetime.now() >= deadline:
            log(f"stop-after {args.stop_after} reached; {len(queue) - i + 1} left for next run")
            break
        prev = db.model_for(str(src))
        size_gb = src.stat().st_size / 1e9
        log(f"[{i}/{len(queue)}] {src}  ({size_gb:.2f} GB{', re-run of ' + prev if prev else ''})")
        duration, has_audio = ffprobe(src)
        if not has_audio:
            db.mark_skipped(str(src), "no audio track")
            log("  skipped: no audio track (won't retry)")
            skipped += 1
            continue
        wav = AUDIO_CACHE / f"{abs(hash(str(src)))}.wav"
        t0 = time.time()
        try:
            digest = sha256_of(src)
            t_hash = time.time() - t0
            extract_audio(src, wav)
            t_extract = time.time() - t0 - t_hash
            result = transcriber.transcribe(
                wav, src.parent / "transcriptions",
                progress_cb=lambda m: log(f"  {m.strip()}"),
                output_name=src.stem,
                meta={"source_path": str(src)},
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  FAILED: {exc}")
            failed += 1
            continue
        finally:
            wav.unlink(missing_ok=True)
        elapsed = time.time() - t0
        dur = duration or result.duration
        audio_s += dur
        rtf = dur / elapsed if elapsed else 0.0
        speech = result.speech_seconds or 0.0
        n_words = sum(len(s["text"].split()) for s in result.segments)
        log(f"  done: {dur / 60:.1f} min audio, speech {speech / 60:.1f} min, "
            f"{n_words} words, {elapsed / 60:.1f} min wall "
            f"(hash {t_hash:.0f}s, extract {t_extract:.0f}s) = {rtf:.1f}x realtime  "
            f"sha256={digest[:12]}…")

        video_id = db.upsert_video(
            path=str(src), filename=src.name, duration=dur,
            language=result.language, model=result.model_name,
        )
        db.insert_segments(video_id, result.segments)
        db.set_sha256(video_id, digest)
        if notion is not None:
            try:
                page_id = notion.push(
                    media_id=str(src), title=src.stem, segments=result.segments,
                    duration=dur, language=result.language, model=result.model_name,
                    file_path=str(src), content_hash=digest,
                )
                db.set_notion_page(video_id, page_id)
                log("  Notion: logged to Media Logs")
            except Exception as exc:  # noqa: BLE001
                log(f"  Notion: FAILED ({exc}) — transcript is saved locally; "
                    "the app's 'Send Existing to Notion' can resend it")
        done += 1

    wall = time.time() - t_run
    pending = len(queue) - done - failed - skipped
    rtf = audio_s / wall if wall else 0.0
    log(f"SUMMARY done={done} failed={failed} no_audio={skipped} pending={pending} "
        f"audio={fmt_h(audio_s)} wall={fmt_h(wall)} realtime={rtf:.1f}x model={args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
