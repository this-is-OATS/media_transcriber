"""Subprocess entry point — runs one transcription, writes JSON result to a file.

Used by the worker so it can SIGTERM mid-file on cancel.

Stdin:   JSON payload (video_path, output_dir, model_name, diarize, hf_token,
                       result_path)
Stderr:  progress lines, prefixed with "PROGRESS: "
Result:  JSON written to `result_path` (so it can't get mixed with stray stdout
         writes from torch/pyannote/etc.)
Exit 0 on success, non-zero on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .transcriber import Transcriber


def main() -> int:
    payload = json.loads(sys.stdin.read())
    transcriber = Transcriber(
        model_name=payload["model_name"],
        diarize=payload.get("diarize", False),
        hf_token=payload.get("hf_token"),
    )

    def emit_progress(msg: str) -> None:
        sys.stderr.write(f"PROGRESS: {msg}\n")
        sys.stderr.flush()

    try:
        result = transcriber.transcribe(
            Path(payload["video_path"]),
            Path(payload["output_dir"]),
            progress_cb=emit_progress,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    out = {
        "video_path": str(result.video_path),
        "markdown_path": str(result.markdown_path),
        "segments": result.segments,
        "duration": result.duration,
        "language": result.language,
        "model_name": result.model_name,
        "diarized": result.diarized,
    }
    Path(payload["result_path"]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
