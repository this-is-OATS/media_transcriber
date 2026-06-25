#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# torchcodec (pulled in by pyannote.audio) wants FFmpeg 4-7 libs.
# brew's `ffmpeg` is currently v8 (libavutil.60); `ffmpeg@7` provides v7 (libavutil.59).
FFMPEG7_LIB="/opt/homebrew/opt/ffmpeg@7/lib"
if [ -d "$FFMPEG7_LIB" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="${FFMPEG7_LIB}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

exec .venv/bin/python -m app.main "$@"
