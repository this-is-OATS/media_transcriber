# Media Transcriber — Setup

## One-time setup (5 minutes)

Run this in Terminal to install all dependencies:

```bash
bash <(curl -s https://raw.githubusercontent.com/this-is-OATS/media_transcriber/main/setup.sh)
```

Or manually:

```bash
# Install Homebrew packages
brew install python@3.11 ffmpeg@7

# Clone the repo
git clone https://github.com/this-is-OATS/media_transcriber.git
cd media_transcriber

# Install Python dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

After setup, just click **Media Transcriber** in Applications, or:

```bash
/Applications/MediaTranscriber.app
```

## First use

When you first use speaker detection (diarization):
1. Get a free Hugging Face token: https://huggingface.co/settings/tokens
2. Accept terms for:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Paste your token into the app's "HF Token" field

Done! Drag videos to transcribe.
