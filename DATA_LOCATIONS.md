# Data Locations

Where every piece of data lives for Media Transcriber and the voice-memo
pipeline. Both write to Notion with **one shared integration** (the
`NOTION_API_KEY` in `~/voice-memo-pipeline/.env`), each into its own database.

## Notion databases

| Database | ID | Written by | Matched on |
|---|---|---|---|
| Daily Field Logs | `5c29346d-d262-4814-858f-d9d11fab4e7d` | voice-memo pipeline (`~/voice-memo-pipeline/notion_logger.py`) | voice memo |
| Media Logs (sister DB) | `eec4d180-8681-4a13-9dad-815b8b6b2286` | Media Transcriber (`app/notion_sync.py`) | `Media ID` property |

Media Logs uses the same Category / Tags / Status properties, "Source
Recording" table and "Notes — Whisper" heading as Daily Field Logs. Re-running
a transcript adds a new section; it never overwrites.

`Media ID` = Apple Photos UUID for Photos videos, full file path for dropped files.

To add another sister database: create it in Notion, add the integration under
••• > Connections, and pass its ID to `NotionSync(token, database_id=...)`.

## Media Transcriber

| What | Location |
|---|---|
| Local database (source of truth: what's been transcribed, Notion page IDs) | `~/Library/Application Support/MediaTranscriber/transcripts.sqlite` |
| Settings (model, diarize, Hugging Face token) | `~/Library/Application Support/MediaTranscriber/settings.json` |
| Bundled runtime | `~/Library/Application Support/MediaTranscriber/runtime/` |
| Logs | `~/Library/Logs/MediaTranscriber/` |
| Transcripts: dropped files/folders | `<dropped folder>/transcriptions/` (or next to a single file) |
| Transcripts: Apple Photos scan | `~/Documents/MediaTranscriber/Apple Photos/` |
| iCloud video downloads (only if "Keep iCloud downloads" is on) | `~/Movies/MediaTranscriber Photos/` |
| Speech models | `~/.cache/whisper/`, `~/.cache/huggingface/` |

Apple Photos library is read only; nothing is written into it.

Current transcript sources: `~/Desktop` and
`/Volumes/Media TxFr/MEDIA BUFFER/...` (external drive).

## Voice-memo pipeline (`~/voice-memo-pipeline`)

| What | Location |
|---|---|
| Config / Notion token | `~/voice-memo-pipeline/.env` |
| Source recordings | `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings` |
| Archive | `/Volumes/NAS/Z_ARCH/ALL APPLE VOICE MEMO ALL TIME` |
| Google Drive copy | `My Drive/APPLE VOICE MEMOS FOR DESCRIPT` (Drive for Desktop mount) |
| Processing state | `processed.json`, `whisper_processed.json`, `summary_processed.json` |
