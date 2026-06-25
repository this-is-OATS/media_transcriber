"""Background transcription worker — runs each file in a subprocess so cancel works mid-file."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
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
    ):
        super().__init__()
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

    def run(self) -> None:
        files: list[Path] = []
        for inp in self.inputs:
            files.extend(find_media_files(inp))
        files = sorted(set(files))

        if not files:
            self.progress.emit("No media files found.")
            self.progress_pct.emit(0, 1)
            self.finished.emit(0, 0)
            return

        self.progress.emit(f"Found {len(files)} media file(s).")
        self.progress_pct.emit(0, len(files))

        success = 0
        fail = 0

        for i, video in enumerate(files, 1):
            if self._cancelled:
                self.progress.emit("Cancelled.")
                break

            self.progress.emit(f"[{i}/{len(files)}] {video.name}")
            if self.db.has_video(str(video)):
                self.progress.emit("  replacing previous transcript")

            self.progress_pct.emit(-1, len(files))

            try:
                result = self._run_one(video)
            except _Cancelled:
                self.progress.emit("  cancelled mid-file")
                break
            except Exception as exc:  # noqa: BLE001
                self.file_failed.emit(str(video), str(exc))
                self.progress.emit(f"  FAILED: {exc}")
                fail += 1
                self.progress_pct.emit(i, len(files))
                continue

            video_id = self.db.upsert_video(
                path=result["video_path"],
                filename=Path(result["video_path"]).name,
                duration=result["duration"],
                language=result["language"],
                model=result["model_name"],
            )
            self.db.insert_segments(video_id, result["segments"])
            self.file_done.emit(result["video_path"])
            success += 1
            self.progress_pct.emit(i, len(files))

        self.finished.emit(success, fail)

    # ----- subprocess plumbing -----
    def _run_one(self, video: Path) -> dict:
        result_fd, result_path = tempfile.mkstemp(prefix="mt_result_", suffix=".json")
        os.close(result_fd)

        payload = {
            "video_path": str(video),
            "output_dir": str(self.output_dir),
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
