"""Background transcription worker — runs each file in a subprocess so cancel works mid-file."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .db import Database
from .transcriber import find_media_files


# torchcodec (pyannote.audio dep) wants FFmpeg 4-7. brew's `ffmpeg` is v8.
# Install `brew install ffmpeg@7` and we'll add it to DYLD_LIBRARY_PATH here.
_FFMPEG7_LIB = "/opt/homebrew/opt/ffmpeg@7/lib"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.path.isdir(_FFMPEG7_LIB):
        # FALLBACK so the `av` package keeps its own bundled libavdevice
        # (avoids "AVFFrameReceiver implemented in both" objc warnings).
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{_FFMPEG7_LIB}:{existing}" if existing else _FFMPEG7_LIB
        )
    return env


def has_audio(path: Path) -> bool:
    """True if the file has an audio stream. On any ffprobe problem, assume yes
    and let the transcriber report the real error."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    return out.returncode != 0 or bool(out.stdout.strip())


@dataclass
class Job:
    label: str                     # shown in the log
    output_dir: Path
    video: Path | None = None      # None until prepared (e.g. iCloud download)
    output_name: str | None = None # markdown filename stem
    meta: dict = field(default_factory=dict)
    db_path: str | None = None     # stable id in the DB (defaults to video path)
    temp_file: bool = False        # delete `video` after transcribing


class TranscriptionWorker(QObject):
    progress = pyqtSignal(str)
    file_done = pyqtSignal(str)
    file_failed = pyqtSignal(str, str)
    finished = pyqtSignal(int, int)  # success_count, fail_count
    # progress_pct(value, maximum). value < 0 means "indeterminate (busy)".
    progress_pct = pyqtSignal(int, int)

    def __init__(
        self,
        inputs: list[Path],
        output_dir: Path,
        db: Database,
        model_name: str = "base",
        diarize: bool = False,
        hf_token: str | None = None,
        notion=None,  # NotionSync | None
    ):
        super().__init__()
        self.notion = notion
        self.inputs = inputs
        self.output_dir = output_dir
        self.db = db
        self.model_name = model_name
        self.diarize = diarize
        self.hf_token = hf_token
        self._cancelled = False
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancelled = True
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

    # ----- hooks for subclasses -----
    def _collect_jobs(self) -> list[Job]:
        files: list[Path] = []
        for inp in self.inputs:
            files.extend(find_media_files(inp))
        return [
            Job(label=f.name, output_dir=self.output_dir, video=f)
            for f in sorted(set(files))
        ]

    def _prepare(self, job: Job) -> None:
        """Make sure job.video exists on disk (subclasses may download it)."""

    # ----- main loop -----
    def run(self) -> None:
        try:
            jobs = self._collect_jobs()
        except Exception as exc:  # noqa: BLE001
            self.progress.emit(f"ERROR: {exc}")
            self.finished.emit(0, 0)
            return

        if not jobs:
            self.progress.emit("Nothing to transcribe.")
            self.progress_pct.emit(0, 1)
            self.finished.emit(0, 0)
            return

        self.progress.emit(f"Found {len(jobs)} media file(s).")
        self.progress_pct.emit(0, len(jobs))

        success = 0
        fail = 0
        skipped = 0

        for i, job in enumerate(jobs, 1):
            if self._cancelled:
                self.progress.emit("Cancelled.")
                break

            self.progress.emit(f"[{i}/{len(jobs)}] {job.label}")
            if job.video is not None and self.db.has_video(job.db_path or str(job.video)):
                self.progress.emit("  replacing previous transcript")

            self.progress_pct.emit(-1, len(jobs))

            try:
                self._prepare(job)
                if not has_audio(job.video):
                    self.db.mark_skipped(
                        job.meta.get("photos_uuid") or job.db_path or str(job.video),
                        "no audio track",
                    )
                    self.progress.emit("  skipped: no audio track (won't retry)")
                    skipped += 1
                    self.progress_pct.emit(i, len(jobs))
                    continue
                result = self._run_one(job)
            except _Cancelled:
                self.progress.emit("  cancelled mid-file")
                break
            except Exception as exc:  # noqa: BLE001
                label = str(job.video or job.label)
                self.file_failed.emit(label, str(exc))
                self.progress.emit(f"  FAILED: {exc}")
                fail += 1
                self.progress_pct.emit(i, len(jobs))
                continue
            finally:
                if job.temp_file and job.video is not None:
                    job.video.unlink(missing_ok=True)
                    try:
                        job.video.parent.rmdir()
                    except OSError:
                        pass

            video_id = self.db.upsert_video(
                path=job.db_path or result["video_path"],
                filename=job.meta.get("original_filename")
                or Path(result["video_path"]).name,
                duration=result["duration"],
                language=result["language"],
                model=result["model_name"],
                meta=job.meta,
            )
            self.db.insert_segments(video_id, result["segments"])
            self._send_to_notion(
                video_id,
                media_id=job.meta.get("photos_uuid") or job.db_path or result["video_path"],
                title=job.output_name or Path(result["video_path"]).stem,
                segments=result["segments"],
                duration=result["duration"],
                language=result["language"],
                model=result["model_name"],
                meta=job.meta,
                file_path=None if (job.db_path or "").startswith("photos://")
                else (job.db_path or result["video_path"]),
            )
            self.file_done.emit(result["video_path"])
            success += 1
            self.progress_pct.emit(i, len(jobs))

        if skipped:
            self.progress.emit(f"Skipped {skipped} with no audio track.")
        self.finished.emit(success, fail)

    def _send_to_notion(self, video_id: int, **page) -> bool:
        """Push one transcript to Notion. Failures are logged, never fatal —
        the transcript is already saved locally and can be re-sent later."""
        if self.notion is None:
            return False
        try:
            page_id = self.notion.push(**page)
        except Exception as exc:  # noqa: BLE001
            self.progress.emit(f"  Notion: FAILED ({exc})")
            return False
        self.db.set_notion_page(video_id, page_id)
        self.progress.emit("  Notion: logged to Media Logs")
        return True

    # ----- subprocess plumbing -----
    def _run_one(self, job: Job) -> dict:
        result_fd, result_path = tempfile.mkstemp(prefix="mt_result_", suffix=".json")
        os.close(result_fd)

        payload = {
            "video_path": str(job.video),
            "output_dir": str(job.output_dir),
            "output_name": job.output_name,
            "meta": job.meta,
            "model_name": self.model_name,
            "diarize": self.diarize,
            "hf_token": self.hf_token,
            "result_path": result_path,
        }
        cmd = [sys.executable, "-m", "app.transcribe_runner"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # transcribe_runner writes nothing to stdout now
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_subprocess_env(),
        )
        with self._lock:
            self._proc = proc

        # Send payload, close stdin so subprocess can start.
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload))
        proc.stdin.close()

        # Pump stderr in a thread so the log updates live.
        stderr_lines: list[str] = []
        def _pump_stderr() -> None:
            assert proc.stderr is not None
            in_torchcodec_block = False
            in_download = False
            for line in proc.stderr:
                line = line.rstrip("\n")
                stderr_lines.append(line)
                # Collapse the torchcodec dlopen traceback into a single line.
                if "torchcodec is not installed correctly" in line:
                    in_torchcodec_block = True
                    self.progress.emit(
                        "  (warning) torchcodec FFmpeg libs missing — "
                        "non-fatal, using whisperx audio loader"
                    )
                    continue
                if in_torchcodec_block:
                    if "[end of libtorchcodec loading traceback]" in line:
                        in_torchcodec_block = False
                    continue
                # Collapse tqdm download bars (e.g. "  12%|██  | 42M/360M ...").
                if "%|" in line and ("MB/s" in line or "B/s" in line):
                    if not in_download:
                        in_download = True
                        self.progress.emit("  downloading model…")
                    continue
                if in_download and ("%|" not in line):
                    in_download = False
                # Drop noisy duplicate-class objc warnings & lightning notice.
                if line.startswith("objc[") and "is implemented in both" in line:
                    continue
                if "Lightning automatically upgraded" in line:
                    continue
                if line.startswith("PROGRESS: "):
                    self.progress.emit(f"  {line[len('PROGRESS: '):]}")
                elif line.strip():
                    self.progress.emit(f"  {line}")
        t = threading.Thread(target=_pump_stderr, daemon=True)
        t.start()

        # Block until subprocess exits.
        proc.wait()
        t.join(timeout=2)

        with self._lock:
            self._proc = None

        if self._cancelled and proc.returncode != 0:
            try:
                os.unlink(result_path)
            except OSError:
                pass
            raise _Cancelled()

        if proc.returncode != 0:
            tail = "\n".join(stderr_lines[-5:]) if stderr_lines else "(no stderr)"
            try:
                os.unlink(result_path)
            except OSError:
                pass
            raise RuntimeError(f"transcribe_runner exited {proc.returncode}: {tail}")

        try:
            text = Path(result_path).read_text(encoding="utf-8")
            return json.loads(text)
        finally:
            try:
                os.unlink(result_path)
            except OSError:
                pass


class _Cancelled(Exception):
    """Internal signal — user cancelled mid-file."""


class NotionBackfillWorker(TranscriptionWorker):
    """Sends transcripts that are in the local database but not yet in Notion."""

    def __init__(self, db: Database, notion):
        super().__init__(inputs=[], output_dir=Path("."), db=db, notion=notion)

    def run(self) -> None:
        rows = self.db.videos_not_in_notion()
        self.progress.emit(f"{len(rows)} transcript(s) not in Notion yet.")
        self.progress_pct.emit(0, max(len(rows), 1))
        ok = fail = 0
        for i, v in enumerate(rows, 1):
            if self._cancelled:
                self.progress.emit("Cancelled.")
                break
            path = v["path"]
            title = (f"{(v['taken_at'] or '')[:10]} {Path(v['filename']).stem}".strip()
                     if v["source"] == "photos" else Path(path).stem)
            self.progress.emit(f"[{i}/{len(rows)}] {title}")
            meta = {k: v[k] for k in ("source", "photos_uuid", "taken_at", "location", "albums")}
            meta["original_filename"] = v["filename"]
            sent = self._send_to_notion(
                v["id"],
                media_id=v["photos_uuid"] or path,
                title=title,
                segments=self.db.segments_for(v["id"]),
                duration=v["duration"] or 0.0,
                language=v["language"],
                model=v["model"],
                meta=meta,
                file_path=None if path.startswith("photos://") else path,
                add_to_existing=False,
            )
            ok += sent
            fail += not sent
            self.progress_pct.emit(i, len(rows))
        self.finished.emit(ok, fail)
