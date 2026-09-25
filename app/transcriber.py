"""Transcription engine — Whisper (no speakers) or WhisperX (with diarization)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import os

import whisper

# Ensure ffmpeg is in PATH (for subprocess spawned from app bundle)
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm",
}


@dataclass
class TranscriptionResult:
    video_path: Path
    markdown_path: Path
    segments: list[dict]   # each: {start, end, text, speaker?}
    duration: float
    language: str
    model_name: str
    diarized: bool
    speech_seconds: float | None = None   # VAD-detected speech (faster-whisper only)


def is_media_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS


def find_media_files(root: Path) -> list[Path]:
    if root.is_file() and is_media_file(root):
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if is_media_file(p))


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segments_to_markdown(
    video_path: Path,
    segments: list[dict],
    language: str,
    model_name: str,
    duration: float,
    diarized: bool,
    title: str | None = None,
    meta: dict | None = None,
    speech_seconds: float | None = None,
) -> str:
    meta = meta or {}
    # batch.py transcribes an extracted audio track; the header should name the
    # video it came from, not the temp .wav.
    source_path = Path(meta["source_path"]) if meta.get("source_path") else video_path
    lines = [f"# {title or source_path.name}", ""]
    if meta.get("source") == "photos":
        lines += [
            f"- **Source:** Apple Photos (`{meta.get('original_filename')}`)",
            f"- **Photos ID:** `{meta.get('photos_uuid')}`",
            f"- **Taken:** {meta.get('taken_at')}",
        ]
        if meta.get("location"):
            lines.append(f"- **Location:** {meta['location']}")
        if meta.get("albums"):
            lines.append(f"- **Albums:** {meta['albums']}")
    else:
        lines.append(f"- **Source:** `{source_path}`")
    lines += [
        f"- **Duration:** {format_timestamp(duration)}",
    ]
    if speech_seconds is not None:
        lines.append(f"- **Speech (VAD):** {format_timestamp(speech_seconds)}")
    lines += [
        f"- **Language:** {language}",
        f"- **Model:** {model_name}",
        f"- **Speakers:** {'yes' if diarized else 'no'}",
        "",
        "---",
        "",
    ]
    last_speaker = None
    for seg in segments:
        ts = format_timestamp(seg["start"])
        text = seg["text"].strip()
        spk = seg.get("speaker")
        if diarized and spk and spk != last_speaker:
            lines.append(f"### {spk}")
            lines.append("")
            last_speaker = spk
        prefix = f"**[{ts}]**"
        if diarized and spk:
            lines.append(f"{prefix} {text}")
        else:
            lines.append(f"{prefix} {text}")
        lines.append("")
    return "\n".join(lines)


class Transcriber:
    """Loads models lazily and reuses them across files.

    If `diarize` is True and `hf_token` is set, uses WhisperX for ASR + alignment
    + speaker diarization. Otherwise falls back to vanilla openai-whisper.

    engine="faster-whisper" (batch.py's choice, Media Transcription Plan Phase 0)
    runs the same open Whisper weights through CTranslate2 with voice-activity
    detection on and no conditioning on the previous segment's text. That pair
    is what stops the "Yeah." once-a-second repetition loop `base` produced on
    IMG_6865 — over noise, plain Whisper feeds its own last line back in as the
    prompt and echoes it until the audio ends.
    """

    def __init__(
        self,
        model_name: str = "base",
        diarize: bool = False,
        hf_token: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        engine: str = "whisper",
        cpu_threads: int | None = None,
    ):
        self.model_name = model_name
        self.diarize = diarize and bool(hf_token)
        self.hf_token = hf_token
        self.device = device
        self.compute_type = compute_type
        self.engine = engine
        self.cpu_threads = cpu_threads
        self._fw_model = None           # faster-whisper model

        self._whisper_model = None      # vanilla whisper model
        self._wx_model = None           # whisperx asr model
        self._wx_align_model = None
        self._wx_align_meta = None
        self._wx_diarize_pipeline = None
        self._wx_align_lang = None

    # ---------- vanilla whisper ----------
    def _load_whisper(self):
        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(self.model_name)
        return self._whisper_model

    def _transcribe_whisper(self, video_path: Path) -> tuple[list[dict], str, float]:
        model = self._load_whisper()
        result = model.transcribe(str(video_path), verbose=False, fp16=False)
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ]
        language = result.get("language", "unknown")
        duration = segments[-1]["end"] if segments else 0.0
        return segments, language, duration

    # ---------- faster-whisper (VAD, no previous-text conditioning) ----------
    FW_BEAM_SIZE = 5
    FW_VAD = {"min_silence_duration_ms": 500}

    def _load_fw(self):
        if self._fw_model is None:
            from faster_whisper import WhisperModel  # noqa: WPS433
            kw = {}
            if self.cpu_threads:
                kw["cpu_threads"] = self.cpu_threads
            self._fw_model = WhisperModel(
                self.model_name, device=self.device,
                compute_type=self.compute_type, **kw,
            )
        return self._fw_model

    def _transcribe_fw(
        self, audio_path: Path, progress_cb: Callable[[str], None] | None
    ) -> tuple[list[dict], str, float, float]:
        model = self._load_fw()
        seg_iter, info = model.transcribe(
            str(audio_path),
            beam_size=self.FW_BEAM_SIZE,
            vad_filter=True,
            vad_parameters=self.FW_VAD,
            condition_on_previous_text=False,
        )
        segments = []
        for s in seg_iter:
            segments.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            if progress_cb and len(segments) % 50 == 0:
                progress_cb(f"  … {format_timestamp(s.end)}")
        duration = float(info.duration or (segments[-1]["end"] if segments else 0.0))
        speech = float(info.duration_after_vad or 0.0)
        return segments, info.language or "unknown", duration, speech

    # ---------- whisperx ----------
    def _load_wx(self):
        if self._wx_model is None:
            import whisperx  # noqa: WPS433
            self._wx_model = whisperx.load_model(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._wx_model

    def _load_wx_align(self, language_code: str):
        import whisperx  # noqa: WPS433
        if self._wx_align_model is None or self._wx_align_lang != language_code:
            self._wx_align_model, self._wx_align_meta = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
            self._wx_align_lang = language_code
        return self._wx_align_model, self._wx_align_meta

    def _load_wx_diarize(self):
        if self._wx_diarize_pipeline is None:
            import inspect
            import whisperx  # noqa: WPS433
            DiarizeCls = getattr(whisperx, "DiarizationPipeline", None) \
                or getattr(whisperx.diarize, "DiarizationPipeline")
            # whisperx renamed `use_auth_token` -> `token` in newer releases.
            params = inspect.signature(DiarizeCls.__init__).parameters
            token_kw = "token" if "token" in params else "use_auth_token"
            self._wx_diarize_pipeline = DiarizeCls(
                **{token_kw: self.hf_token}, device=self.device
            )
        return self._wx_diarize_pipeline

    def _transcribe_wx(
        self, video_path: Path, progress_cb: Callable[[str], None] | None
    ) -> tuple[list[dict], str, float]:
        import whisperx  # noqa: WPS433

        if progress_cb:
            progress_cb("Loading audio…")
        audio = whisperx.load_audio(str(video_path))

        if progress_cb:
            progress_cb("Transcribing (whisperx)…")
        asr = self._load_wx()
        result = asr.transcribe(audio, batch_size=8)
        language = result["language"]

        if progress_cb:
            progress_cb("Aligning word timestamps…")
        align_model, align_meta = self._load_wx_align(language)
        aligned = whisperx.align(
            result["segments"], align_model, align_meta, audio,
            device=self.device, return_char_alignments=False,
        )

        if progress_cb:
            progress_cb("Diarizing speakers…")
        diarize = self._load_wx_diarize()
        diarize_segments = diarize(audio)
        with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)

        segments = []
        for s in with_speakers["segments"]:
            segments.append({
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s.get("text", ""),
                "speaker": s.get("speaker"),
            })
        duration = segments[-1]["end"] if segments else 0.0
        return segments, language, duration

    # ---------- main entry ----------
    def transcribe(
        self,
        video_path: Path,
        output_dir: Path,
        progress_cb: Callable[[str], None] | None = None,
        output_name: str | None = None,
        meta: dict | None = None,
    ) -> TranscriptionResult:
        speech_seconds: float | None = None
        if progress_cb:
            mode = ("whisperx + diarization" if self.diarize
                    else "faster-whisper (vad, no prev-text)" if self.engine == "faster-whisper"
                    else "whisper")
            progress_cb(f"Mode: {mode}, model: {self.model_name}")

        if self.diarize:
            segments, language, duration = self._transcribe_wx(video_path, progress_cb)
        elif self.engine == "faster-whisper":
            if progress_cb:
                progress_cb("Transcribing…")
            segments, language, duration, speech_seconds = self._transcribe_fw(
                video_path, progress_cb)
        else:
            if progress_cb:
                progress_cb("Transcribing…")
            segments, language, duration = self._transcribe_whisper(video_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{output_name or video_path.stem}.md"
        md_path.write_text(
            segments_to_markdown(
                video_path, segments, language, self.model_name, duration,
                diarized=self.diarize, title=output_name, meta=meta,
                speech_seconds=speech_seconds,
            ),
            encoding="utf-8",
        )

        if progress_cb:
            progress_cb(f"Saved: {md_path.name}")

        return TranscriptionResult(
            video_path=video_path,
            markdown_path=md_path,
            segments=segments,
            duration=duration,
            language=language,
            model_name=self.model_name,
            diarized=self.diarize,
            speech_seconds=speech_seconds,
        )
