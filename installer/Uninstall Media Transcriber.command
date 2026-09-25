#!/usr/bin/env bash
# Double-click to remove Media Transcriber. Your transcripts (.md files) are NOT touched.
set -e

APP="/Applications/Media Transcriber.app"
DATA="$HOME/Library/Application Support/MediaTranscriber"

echo "This removes:"
echo "  $APP"
echo "  $DATA/runtime  (app code + Python environment)"
echo
read -r -p "Also delete the search database and saved settings? [y/N] " wipe
read -r -p "Continue? [y/N] " ok
[ "$ok" = "y" ] || [ "$ok" = "Y" ] || { echo "Cancelled."; exit 0; }

rm -rf "$APP" "$DATA/runtime"
if [ "$wipe" = "y" ] || [ "$wipe" = "Y" ]; then
    rm -rf "$DATA"
fi
echo "✅ Uninstalled."
