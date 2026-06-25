"""Media Transcriber — PyQt6 desktop app.

Drag-and-drop media files (or a folder) and they get transcribed via Whisper
(or WhisperX with speaker diarization) into timestamped markdown files.
Each transcript is also indexed in SQLite for full-text search.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .db import Database
from .worker import TranscriptionWorker


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "base"

APP_DATA = Path.home() / "Library" / "Application Support" / "MediaTranscriber"
SETTINGS_PATH = APP_DATA / "settings.json"


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


class DropZone(QLabel):
    """Big drop area that accepts files and folders."""

    def __init__(self, on_drop):
        super().__init__()
        self._on_drop = on_drop
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(160)
        self.setText(
            "Drop media files or a folder here\n\n"
            "(or use 'Choose Files…' / 'Choose Folder…' below)"
        )
        self._reset_style()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_active_style()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._reset_style()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
        paths = [p for p in paths if p.exists()]
        self._reset_style()
        if paths:
            self._on_drop(paths)

    def _reset_style(self) -> None:
        self.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 12px; "
            "color: #666; font-size: 14px; padding: 24px; }"
        )

    def _set_active_style(self) -> None:
        self.setStyleSheet(
            "QLabel { border: 2px dashed #3a8; border-radius: 12px; "
            "color: #3a8; font-size: 14px; padding: 24px; }"
        )


class MainWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.setWindowTitle("Media Transcriber")
        self.resize(860, 680)

        self.db = db
        self.settings = load_settings()
        self._thread: QThread | None = None
        self._worker: TranscriptionWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.drop_zone = DropZone(on_drop=self.handle_paths)
        layout.addWidget(self.drop_zone)

        # File pickers + model selector
        controls = QHBoxLayout()
        self.btn_files = QPushButton("Choose Files…")
        self.btn_files.clicked.connect(self.choose_files)
        controls.addWidget(self.btn_files)

        self.btn_folder = QPushButton("Choose Folder…")
        self.btn_folder.clicked.connect(self.choose_folder)
        controls.addWidget(self.btn_folder)

        controls.addStretch()
        controls.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        self.model_combo.setCurrentText(self.settings.get("model", DEFAULT_MODEL))
        controls.addWidget(self.model_combo)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_transcription)
        controls.addWidget(self.btn_cancel)
        layout.addLayout(controls)

        # Diarization row
        diar_row = QHBoxLayout()
        self.chk_diarize = QCheckBox("Detect speakers (WhisperX)")
        self.chk_diarize.setChecked(self.settings.get("diarize", False))
        self.chk_diarize.toggled.connect(self._on_diarize_toggled)
        diar_row.addWidget(self.chk_diarize)

        diar_row.addWidget(QLabel("HF Token:"))
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setPlaceholderText("hf_… (saved locally)")
        self.hf_token_input.setText(self.settings.get("hf_token", ""))
        self.hf_token_input.setEnabled(self.chk_diarize.isChecked())
        diar_row.addWidget(self.hf_token_input, 1)
        layout.addLayout(diar_row)

        # First-time hint for HF
        self.hf_hint = QLabel(
            "Speaker detection needs a free Hugging Face token + accepted terms for "
            "<a href='https://huggingface.co/pyannote/speaker-diarization-community-1'>pyannote/speaker-diarization-community-1</a> "
            "and <a href='https://huggingface.co/pyannote/segmentation-3.0'>pyannote/segmentation-3.0</a>."
        )
        self.hf_hint.setOpenExternalLinks(True)
        self.hf_hint.setWordWrap(True)
        self.hf_hint.setStyleSheet("color: #888; font-size: 11px;")
        self.hf_hint.setVisible(self.chk_diarize.isChecked())
        layout.addWidget(self.hf_hint)

        # Output dir display
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self.out_label = QLabel("(picked when you drop files)")
        self.out_label.setStyleSheet("color: #888;")
        out_row.addWidget(self.out_label, 1)
        layout.addLayout(out_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)

        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Activity log will appear here…")
        layout.addWidget(self.log, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")

    # ----- settings helpers -----
    def _on_diarize_toggled(self, on: bool) -> None:
        self.hf_token_input.setEnabled(on)
        self.hf_hint.setVisible(on)

    def _persist_settings(self) -> None:
        self.settings.update({
            "model": self.model_combo.currentText(),
            "diarize": self.chk_diarize.isChecked(),
            "hf_token": self.hf_token_input.text().strip(),
        })
        save_settings(self.settings)

    # ----- file pickers -----
    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose media files",
            str(Path.home()),
            "Media (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac "
            "*.mp4 *.mov *.m4v *.mkv *.avi *.webm)",
        )
        if files:
            self.handle_paths([Path(f) for f in files])

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder", str(Path.home())
        )
        if folder:
            self.handle_paths([Path(folder)])

    # ----- transcription -----
    def handle_paths(self, paths: list[Path]) -> None:
        if self._thread is not None:
            self.log_line("Already transcribing — wait or cancel first.")
            return

        diarize = self.chk_diarize.isChecked()
        hf_token = self.hf_token_input.text().strip()
        if diarize and not hf_token:
            self.log_line("ERROR: speaker detection is on but no HF token set.")
            return
        self._persist_settings()

        # Output dir = "transcriptions" inside dropped folder OR next to the file.
        first = paths[0]
        anchor = first if first.is_dir() else first.parent
        output_dir = anchor / "transcriptions"

        self.out_label.setText(str(output_dir))
        self.log_line(f"Output dir: {output_dir}")

        self._thread = QThread(self)
        self._worker = TranscriptionWorker(
            inputs=paths,
            output_dir=output_dir,
            db=self.db,
            model_name=self.model_combo.currentText(),
            diarize=diarize,
            hf_token=hf_token or None,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress_msg)
        self._worker.progress_pct.connect(self._on_progress_pct)
        self._worker.finished.connect(self._on_finished)

        self._set_busy(True)
        self.statusBar().showMessage("Processing…")
        self._thread.start()

    def cancel_transcription(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log_line("Cancellation requested…")

    def _on_progress_msg(self, msg: str) -> None:
        self.log_line(msg)
        # Top-level lines like "[1/3] foo.mov" become the status bar text.
        stripped = msg.lstrip()
        if stripped.startswith("[") and "]" in stripped and msg == stripped:
            self.statusBar().showMessage(stripped)

    def _on_progress_pct(self, value: int, maximum: int) -> None:
        if value < 0:
            # Indeterminate / busy
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Transcribing…")
        else:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(value)
            self.progress_bar.setFormat(f"{value} / {maximum} files")

    def _on_finished(self, success: int, fail: int) -> None:
        self.log_line(f"Done. {success} succeeded, {fail} failed.")
        self.statusBar().showMessage(f"Finished — {success} ok, {fail} failed.")
        # Make sure the bar lands at 100%, then resets to "Idle".
        total = success + fail
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(total)
            self.progress_bar.setFormat(f"Done — {success} ok, {fail} failed")
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Idle")
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.btn_files.setEnabled(not busy)
        self.btn_folder.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.chk_diarize.setEnabled(not busy)
        self.hf_token_input.setEnabled(not busy and self.chk_diarize.isChecked())
        self.btn_cancel.setEnabled(busy)

    def log_line(self, msg: str) -> None:
        self.log.appendPlainText(msg)


def main() -> int:
    db = Database(APP_DATA / "transcripts.sqlite")
    app = QApplication(sys.argv)
    win = MainWindow(db=db)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
