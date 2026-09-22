"""MEDIA_TRANSCRIBER — local media transcription with optional speaker diarization."""

__version__ = "0.3.2"


def version_label() -> str:
    """Version plus the time the code was last changed, so a stale running
    copy is obvious (compare with a fresh launch)."""
    from datetime import datetime
    from pathlib import Path
    newest = max(f.stat().st_mtime for f in Path(__file__).parent.glob("*.py"))
    return f"v{__version__} · code {datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}"
