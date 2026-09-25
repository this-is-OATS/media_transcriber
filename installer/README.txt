MEDIA TRANSCRIBER
=================

Drag video/audio files onto the app and get timestamped Markdown transcripts,
saved in a "transcriptions" folder next to the originals.

REQUIREMENTS
  - Mac with Apple Silicon (M1 or newer), macOS 12+
  - ~5 GB free disk space (Python, PyTorch, speech models)
  - Internet connection for the first install

INSTALL
  1. Double-click "Install Media Transcriber.command"
     - If macOS blocks it: right-click the file > Open > Open.
  2. A Terminal window runs the setup (5-15 minutes the first time).
  3. Open "Media Transcriber" from your Applications folder.

SPEAKER DETECTION (optional)
  1. Create a free token at https://huggingface.co/settings/tokens (Read access)
  2. While logged in, click "Agree" on:
       https://huggingface.co/pyannote/speaker-diarization-community-1
       https://huggingface.co/pyannote/segmentation-3.0
  3. Paste the token (starts with "hf_") into the app's HF Token field.

APPLE PHOTOS
  Click "Scan Photos Library" to transcribe videos from Apple Photos.
  - Your library is only read, never changed.
  - The first time, macOS asks to allow Photos access: click Allow.
  - Videos stored in iCloud are downloaded one at a time, transcribed, then
    deleted (tick "Keep iCloud downloads" to keep them in ~/Movies).
  - Already-transcribed videos are skipped, so each run picks up where the
    last one stopped. Use "max videos per run" to work in batches.
  - Transcripts: ~/Documents/MediaTranscriber/Apple Photos/
    named like "2023-04-23 IMG_0766.md", with date, place and albums.

UPDATING
  Run the installer again from a newer copy of this folder.

UNINSTALL
  Double-click "Uninstall Media Transcriber.command".
  Your transcript .md files are never deleted.
