#!/usr/bin/env bash
set -e

echo "Creating Media Transcriber.app…"
echo ""

# Find the run.sh script path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "ERROR: run.sh not found at $RUN_SCRIPT"
    exit 1
fi

# Create the app using Platypus
platypus \
    -a "Media Transcriber" \
    -o "Text Window" \
    -p /bin/bash \
    -c "$RUN_SCRIPT" \
    -R \
    "/Applications/MediaTranscriber.app"

echo ""
echo "✅ App created at /Applications/MediaTranscriber.app"
echo ""
echo "You can now:"
echo "  - Click the app in Applications folder, OR"
echo "  - Run: open /Applications/MediaTranscriber.app"
