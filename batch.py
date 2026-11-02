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
        --from-manifest CSV       after the roots, queue every NAS video in
                                  voice-memo-pipeline's media_manifest.csv,
                                  smallest first (video extensions only — a
                                  bare /Volumes/NAS root would sweep in music
                                  and the voice-memo archive)
        --skip-bucket PREFIX      leave out manifest rows in that bucket (repeatable)
        --screen-recordings last|skip|inline
                                  where screen recordings go in the queue;
                                  default last — see below
        --output-dir DIR          write markdown here as <stem>__<sha12>.md
                                  instead of a transcriptions/ folder beside
                                  each video (the whole-NAS run would litter
                                  hundreds of folders otherwise)

Screen recordings and screenshots go LAST (v0.7.3, Johnny 2026-11-02; the
problem is from the 2026-10-19 memo, "fix media transcriber problem with
scanning screenshots and sorting"). A screen recording is a movie with an audio
track, so it was queued like any other video — and because they are small they
sorted to the FRONT of the smallest-first queue, so phone-UI narration went
through the model ahead of real footage. v0.7.2 left them out; 0.7.3 keeps them
in the queue and puts them BEHIND every other video instead, each group still
smallest-first, so they are transcribed eventually and nothing is written off.
They are recognised by the manifest's own `screen_recording` column
(voice-memo-pipeline 1.4.96+; it was `kind` in 1.4.91-1.4.95 and both are still
read, since the name collided with that repo's unrelated `Music Kind`), or by
the names the capture tools give them and a folder named for them. `--screen-recordings skip` is 0.7.2's behaviour and `inline` is pre-0.7.2.
Only this headless runner reorders: a file dropped on the app's window is a
deliberate choice and runs as asked.

v0.7.4, measured against the real NAS (1,678 of 34,247 manifest rows): a folder
whose name says SELFIE, TRANSCRIBE, "Live Journals", VLOG or DOGCAST (the last
two added in 0.7.5) wins outright and keeps its place in the queue, ahead of
even the manifest's own kind column (0.7.6 — that column is only as new as the
last rebuild, and the wrapper runs on the previous manifest when one fails), because a screen recording filed under LONG SELFIE
TRANSCRIBE is Johnny talking with his phone screen going. The same run showed
SCREEN CAPS and SCREENCAPS folders were missed by "screen capture", so the
pattern is now "screen cap"/"screencap".

Guards for the whole-NAS run: a file whose container claims more than 12 hours
is skipped as broken (one 0 MB file reports 596 h — its audio would not fit the
boot drive); a file whose extracted audio would not leave 2 GB free is left for
a later night; a file whose SHA-256 is already transcribed under another path
is recorded as a duplicate rather than transcribed twice.

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
import csv
import hashlib
import shutil
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
MAX_PLAUSIBLE_S = 12 * 3600          # longer than this = broken container
WAV_BYTES_PER_S = 32000              # 16 kHz mono s16
MIN_FREE_BYTES = 2 * 1024 ** 3

# What a capture tool calls a screen recording or a screenshot, in the file's
# own name or in a folder named for them. Matched against the name lowercased
# with "_" and "-" read as spaces, so "Screen_Recording", "screen-recording"
# and "Screen Recording" are one pattern. Kept in step with
# build_media_manifest.py in voice-memo-pipeline, which writes the manifest's
# `kind` column with the same list — change both together.
SCREEN_PATTERNS = (
    "screen recording",      # iOS/macOS, and the SCREEN RECORDINGS folders
    "screenrecording",
    "screen shot", "screenshot",
    "screen cap", "screencap",   # v0.7.4: also the SCREEN CAPS and SCREENCAPS
                                 # folders, which "screen capture" missed — the
                                 # 2026-11-02 --list-screen run over the real
                                 # NAS found files in them caught only by their
                                 # own names
    "screencast",
    "rpreplay",              # iOS ReplayKit: RPReplay_Final1698275632.mov
    "simulator screen",      # Xcode Simulator captures
)

# Folders whose name says this is content Johnny wants transcribed, whatever the
# file is called (v0.7.4, his call on 2026-11-02 after reading the real list:
# "the folder wins"). A screen recording under LONG SELFIE TRANSCRIBE is him
# talking with his phone screen going — he filed it there on purpose, so it
# keeps its place in the queue instead of going behind everything else. Matched
# on folder names only, never the file's own name. Kept in step with
# build_media_manifest.py in voice-memo-pipeline — change both together.
KEEP_PATTERNS = (
    "selfie",                # SELFIE, KENOSHA SELFIES, SELFIE CAM, …
    "transcribe",            # LONG SELFIE TRANSCRIBE, JUST TRANSCRIBE
    "live journal",          # SORTING AREA/Live Journals
    "vlog",                  # v0.7.5, Johnny 2026-11-02
    "dogcast",               # the podcast — DOGCAST WITH WATERS ABOVE, whose
                             # source clips are RPReplay screen recordings
)

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


def is_screen_name(name: str) -> bool:
    flat = name.lower().replace("_", " ").replace("-", " ")
    return any(p in flat for p in SCREEN_PATTERNS)


def is_keep_path(path: Path) -> bool:
    """A KEEP_PATTERNS folder anywhere below the volume.

    Read from the whole path, not just below the root: the wrapper queues
    ".../LONG SELFIE TRANSCRIBE" and ".../SELFIE" as roots themselves, and
    relative_to() would hide exactly the folder whose name is the point.
    """
    return any(any(k in p.lower() for k in KEEP_PATTERNS) for p in path.parent.parts[3:])


def is_screen_recording(path: Path, root: Path | None = None) -> bool:
    """A screen recording or screenshot, by its own name or by its folder.

    A folder named for them counts: a SCREEN RECORDINGS folder full of files
    named IMG_0042.mov is still a folder of screen recordings. Only the folders
    BELOW the root are read when a root is given, so a drop folder that is
    itself called "SCREENSHOTS" does not tag everything the caller asked for —
    naming a root is a deliberate choice, the same rule as the app's window.

    A KEEP_PATTERNS folder wins outright (v0.7.4): what Johnny filed under
    SELFIE or LONG SELFIE TRANSCRIBE is him talking, and keeps its place in the
    queue however the capture tool named the file.
    """
    if is_keep_path(path):
        return False
    if is_screen_name(path.name):
        return True
    parts = path.parent.parts[3:]    # skip "/", "Volumes", "<volume>"
    if root is not None:
        try:
            parts = path.parent.relative_to(root).parts
        except ValueError:
            pass
    return any(is_screen_name(p) for p in parts)


def collect(roots: list[Path], mark_screen: bool = True) -> list[Path]:
    """Roots in the order given (smallest bucket first is the caller's job);
    inside a root, smallest file first — the same rule one level down.

    Nothing is left out here. A screen recording is recorded in SCREEN_PATHS
    and the queue order decides what happens to it (v0.7.3); mark_screen=False
    is --screen-recordings inline, where they are ordinary videos."""
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            log(f"WARNING: root missing, skipped: {root}")
            continue
        batch = [f for f in find_media_files(root) if f not in seen]
        batch.sort(key=lambda f: f.stat().st_size)
        if mark_screen:
            SCREEN_PATHS.update(str(f) for f in batch if is_screen_recording(f, root))
        files.extend(batch)
        seen.update(batch)
    return files


DURATIONS: dict[str, float] = {}   # path -> seconds, from the manifest (hours remaining)
# Paths this run treats as screen recordings: filled by collect() (name or a
# folder below the root) and by manifest_files() (the manifest's own `kind`
# column, with the same name/folder match as the fallback). The queue order
# reads it — see order_queue().
SCREEN_PATHS: set[str] = set()


def manifest_files(csv_path: Path, skip_buckets: list[str] = (),
                   mark_screen: bool = True) -> list[Path]:
    """NAS videos from media_manifest.csv, smallest first. Drops Photos-only
    rows, duplicates the manifest already knows, files with no audio stream,
    containers claiming an implausible duration, and any bucket named in
    skip_buckets (prefix match on the manifest's bucket column — e.g. the
    Live Photo clips under "PHOTOS EXPORT/PHOTOS", Johnny 2026-10-22).

    Screen recordings are queued like everything else and recorded in
    SCREEN_PATHS for the queue order (v0.7.3): the manifest's own
    screen_recording column when it has one (or `kind`, its 1.4.91-1.4.95
    name), otherwise this runner's name and folder match, so an older manifest
    is classified here rather than trusted."""
    rows = []
    dropped = skipped_b = screen = 0
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source"] != "nas" or r["dupe_of"] or r["has_audio"] == "False":
                dropped += r["source"] == "nas"
                continue
            if any(r["bucket"].startswith(b) for b in skip_buckets):
                skipped_b += 1
                continue
            if r["duration_s"] and float(r["duration_s"]) > MAX_PLAUSIBLE_S:
                dropped += 1
                continue
            # KEEP is read here too, AHEAD of the kind column (v0.7.6). That
            # column is only as new as the last manifest rebuild, and the
            # wrapper deliberately runs on the previous manifest when a rebuild
            # fails — so a row written before a keep word existed would stay
            # demoted, silently, for as long as the rebuilds keep failing. This
            # runner's own list is the newer of the two; let it win.
            if mark_screen and not is_keep_path(Path(r["path"])) and (
                    r.get("screen_recording") == "True" or r.get("kind") == "screen"
                    or is_screen_recording(Path(r["path"]))):
                SCREEN_PATHS.add(r["path"])
                screen += 1
            rows.append((int(r["size_bytes"] or 0), r["path"]))
            if r["duration_s"]:
                DURATIONS[r["path"]] = float(r["duration_s"])
    rows.sort()
    log(f"manifest {csv_path.name}: {len(rows)} NAS videos queued, {dropped} dropped "
        f"(no audio stream / duplicate / implausible duration), {skipped_b} in skipped buckets, "
        f"{screen} screen recording(s)")
    return [Path(p) for _, p in rows]


def order_queue(queue: list[Path], policy: str) -> list[Path]:
    """The queue with the screen recordings where the policy puts them.

    "last" (the default, Johnny 2026-11-02): they stay in the queue but go
    behind every other video, each group still smallest-first. Real footage is
    never waiting behind phone-UI narration again, and nothing is written off —
    when the backlog of footage is done, these run.
    "skip" is v0.7.2's behaviour, "inline" is pre-0.7.2 (no special handling).
    """
    if policy == "inline" or not SCREEN_PATHS:
        return queue
    head = [f for f in queue if str(f) not in SCREEN_PATHS]
    tail = [f for f in queue if str(f) in SCREEN_PATHS]
    if not tail:
        return queue
    if policy == "skip":
        log(f"{len(tail)} screen recording(s) / screenshot(s) left out of the queue "
            f"(--screen-recordings skip)")
        return head
    log(f"{len(tail)} screen recording(s) / screenshot(s) moved behind the "
        f"{len(head)} other video(s) (--screen-recordings last)")
    return head + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", type=Path)
    ap.add_argument("--from-manifest", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--skip-bucket", action="append", default=[], metavar="PREFIX",
                    help="leave out manifest rows whose bucket starts with PREFIX (repeatable)")
    ap.add_argument("--screen-recordings", choices=("last", "skip", "inline"), default="last",
                    help="where screen recordings and screenshots go in the queue: "
                         "last (default — behind every other video), skip (left out), "
                         "inline (no special handling)")
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
    mark_screen = args.screen_recordings != "inline"
    files = collect(args.roots, mark_screen)
    if args.from_manifest:
        seen = set(files)
        files += [f for f in manifest_files(args.from_manifest, args.skip_bucket,
                                           mark_screen) if f not in seen]
    if not files:
        log("nothing queued (no roots, no manifest)")
        return 0

    queue, already, noaudio_known = [], 0, 0
    for f in files:
        key = str(f)
        if db.is_skipped(key):
            noaudio_known += 1
        elif db.model_for(key) == args.model:
            already += 1
        else:
            queue.append(f)
    # The screen recordings go to the back here, after the already-done ones
    # are out, so "pending" counts what will actually run tonight.
    queue = order_queue(queue, args.screen_recordings)
    log(f"model={args.model}  found={len(files)}  done_with_model={already}  "
        f"known_no_audio={noaudio_known}  pending={len(queue)}")
    if args.pending:
        print(f"PENDING {len(queue)}")
        return 0
    if args.dry_run:
        for f in queue[:40]:
            prev = db.model_for(str(f))
            log(f"  would run: {f}{'  (re-run: was ' + prev + ')' if prev else ''}")
        if len(queue) > 40:
            log(f"  … and {len(queue) - 40} more")
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

    done = failed = skipped = dupes = missing = 0
    audio_s = 0.0
    t_run = time.time()
    for i, src in enumerate(queue, 1):
        if args.limit and done + failed + skipped + dupes >= args.limit:
            log(f"limit {args.limit} reached")
            break
        if deadline and datetime.now() >= deadline:
            log(f"stop-after {args.stop_after} reached; {len(queue) - i + 1} left for next run")
            break
        if not src.exists():
            log(f"[{i}/{len(queue)}] gone since the manifest was built: {src}")
            missing += 1
            continue
        prev = db.model_for(str(src))
        size_gb = src.stat().st_size / 1e9
        log(f"[{i}/{len(queue)}] {src}  ({size_gb:.2f} GB{', re-run of ' + prev if prev else ''})")
        duration, has_audio = ffprobe(src)
        if not has_audio:
            db.mark_skipped(str(src), "no audio track")
            log("  skipped: no audio track (won't retry)")
            skipped += 1
            continue
        if duration and duration > MAX_PLAUSIBLE_S:
            db.mark_skipped(str(src), f"implausible duration {duration / 3600:.0f} h")
            log(f"  skipped: container claims {duration / 3600:.0f} h — broken file (won't retry)")
            skipped += 1
            continue
        AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
        need = (duration or 3600) * WAV_BYTES_PER_S + MIN_FREE_BYTES
        if shutil.disk_usage(AUDIO_CACHE).free < need:
            log(f"  deferred: boot drive lacks room for {(duration or 3600) / 3600:.1f} h of audio "
                f"(+2 GB margin) — left for a later night")
            failed += 1
            continue
        wav = AUDIO_CACHE / f"{abs(hash(str(src)))}.wav"
        t0 = time.time()
        try:
            digest = sha256_of(src)
            t_hash = time.time() - t0
            twin = db.path_for_sha256(digest, args.model)
            if twin and twin != str(src):
                db.mark_skipped(str(src), f"duplicate of {twin} ({digest})")
                log(f"  duplicate content of {twin} — not transcribed twice")
                dupes += 1
                continue
            extract_audio(src, wav)
            t_extract = time.time() - t0 - t_hash
            result = transcriber.transcribe(
                wav, args.output_dir or src.parent / "transcriptions",
                progress_cb=lambda m: log(f"  {m.strip()}"),
                output_name=f"{src.stem}__{digest[:12]}" if args.output_dir else src.stem,
                meta={"source_path": str(src)},
            )
        except subprocess.CalledProcessError:
            # ffmpeg could not read it (truncated download, 0-byte stub, not
            # really a video). That will not change tomorrow — never retry.
            db.mark_skipped(str(src), "unreadable: ffmpeg could not extract audio")
            log("  skipped: ffmpeg cannot read this file (won't retry)")
            skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001
            log(f"  FAILED: {str(exc)[:300]}")
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
    pending = len(queue) - done - skipped - dupes - missing
    # Hours still to transcribe, for the morning report: manifest durations of
    # what this run did not reach (files not in the manifest count as unknown).
    left = queue[done + failed + skipped + dupes + missing:]
    remaining_s = sum(DURATIONS.get(str(p), 0.0) for p in left)
    unknown = sum(1 for p in left if str(p) not in DURATIONS)
    rtf = audio_s / wall if wall else 0.0
    log(f"SUMMARY done={done} failed={failed} skipped={skipped} dupes={dupes} "
        f"gone={missing} pending={pending} remaining={fmt_h(remaining_s)}"
        f"{'+' + str(unknown) + 'unknown' if unknown else ''} "
        f"audio={fmt_h(audio_s)} wall={fmt_h(wall)} realtime={rtf:.1f}x model={args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
