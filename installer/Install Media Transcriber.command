#!/usr/bin/env bash
# Double-click to install Media Transcriber.
set -e

SRC="$(cd "$(dirname "$0")" && pwd)"
RUNTIME="$HOME/Library/Application Support/MediaTranscriber/runtime"
APP="/Applications/Media Transcriber.app"

echo "=== Media Transcriber installer ==="
echo

# 1. Homebrew
if ! command -v brew >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/brew ]; then
    echo "Homebrew is required. Installing it now (you may be asked for your password)…"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"

# 2. System dependencies
echo "Installing Python 3.11 and FFmpeg…"
brew install python@3.11 ffmpeg ffmpeg@7

# 3. App code
echo "Copying app files to: $RUNTIME"
mkdir -p "$RUNTIME"
rm -rf "$RUNTIME/app"
cp -R "$SRC/payload/app" "$RUNTIME/app"
cp "$SRC/payload/requirements.txt" "$SRC/payload/run.sh" "$RUNTIME/"
chmod +x "$RUNTIME/run.sh"

# 4. Python environment (large download the first time: torch, whisper, etc.)
echo "Setting up Python environment (this can take 5-10 minutes)…"
if [ ! -x "$RUNTIME/.venv/bin/python" ]; then
    "$(brew --prefix python@3.11)/bin/python3.11" -m venv "$RUNTIME/.venv"
fi
"$RUNTIME/.venv/bin/pip" install --upgrade pip
"$RUNTIME/.venv/bin/pip" install -r "$RUNTIME/requirements.txt"

# 5. The .app bundle
echo "Creating $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$SRC/payload/icon.icns" "$APP/Contents/Resources/icon.icns"
cat > "$APP/Contents/MacOS/launch" <<EOF
#!/bin/bash
exec "$RUNTIME/run.sh"
EOF
chmod +x "$APP/Contents/MacOS/launch"
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>launch</string>
    <key>CFBundleName</key><string>Media Transcriber</string>
    <key>CFBundleDisplayName</key><string>Media Transcriber</string>
    <key>CFBundleIdentifier</key><string>com.oats.mediatranscriber</string>
    <key>CFBundleVersion</key><string>0.3.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSPhotoLibraryUsageDescription</key><string>Media Transcriber reads your videos (read-only) to transcribe them. Nothing in your library is changed.</string>
    <key>NSAppleEventsUsageDescription</key><string>Media Transcriber asks Photos to export videos stored in iCloud so they can be transcribed.</string>
</dict>
</plist>
EOF
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
touch "$APP"

echo
echo "✅ Installed. Open 'Media Transcriber' from your Applications folder."
echo "   (Speaker detection needs a free Hugging Face token — see README.)"
open -R "$APP"
