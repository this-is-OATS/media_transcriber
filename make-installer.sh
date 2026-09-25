#!/usr/bin/env bash
# Builds a shareable installer folder (and a .zip of it) from this repo.
# Usage: ./make-installer.sh [output_dir]
set -e
cd "$(dirname "$0")"

OUT="${1:-../MEDIA_TRANSCRIBER_APP}"
rm -rf "$OUT"
mkdir -p "$OUT/payload"

cp installer/*.command installer/README.txt "$OUT/"
chmod +x "$OUT"/*.command

rsync -a --exclude '__pycache__' app "$OUT/payload/"
cp requirements.txt run.sh installer/icon.icns "$OUT/payload/"

(cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && \
    ditto -c -k --keepParent "$(basename "$OUT")" "$(basename "$OUT").zip")

echo "Built: $OUT"
echo "Zip:   $OUT.zip"
