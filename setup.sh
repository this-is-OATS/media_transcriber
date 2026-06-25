#!/usr/bin/env bash
set -e

echo "Installing Media Transcriber dependencies…"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew…"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "Installing Python 3.11 and FFmpeg 7…"
brew install python@3.11 ffmpeg@7

echo ""
echo "Creating Python virtual environment…"
python3.11 -m venv .venv
source .venv/bin/activate

echo "Installing Python packages…"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next: Run the app with:"
echo "  cd $(pwd)"
echo "  ./run.sh"
