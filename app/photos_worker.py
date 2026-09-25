"""Worker that transcribes every not-yet-done video in the Apple Photos library."""
from __future__ import annotations

from pathlib import Path

from . import photos_source
from .db import Database
from .worker import Job, TranscriptionWorker


class PhotosScanWorker(TranscriptionWorker):
    def __init__(
        self,
        output_dir: Path,
        cache_dir: Path,
        db: Database,
        keep_downloads: bool = False,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(inputs=[], output_dir=output_dir, db=db, **kwargs)
        self.cache_dir = cache_dir
        self.keep_downloads = keep_downloads
        self.limit = limit

    def _collect_jobs(self) -> list[Job]:
        self.progress.emit("Reading Photos library (read-only)…")
        videos = photos_source.list_videos()
        done = self.db.transcribed_photos_uuids()
        todo = [v for v in videos if v.uuid not in done]
        in_icloud = sum(1 for v in todo if not v.local_path)
        self.progress.emit(
            f"Library has {len(videos)} videos — {len(videos) - len(todo)} already "
            f"transcribed, {len(todo)} to do ({in_icloud} must download from iCloud)."
        )
        if self.limit:
            todo = todo[: self.limit]
            self.progress.emit(f"Doing the first {len(todo)} this run.")

        jobs = []
        for v in todo:
            jobs.append(Job(
                label=f"{v.output_name}  ({v.location or 'no location'})",
                output_dir=self.output_dir,
                video=Path(v.local_path) if v.local_path else None,
                output_name=v.output_name,
                meta=v.meta(),
                db_path=v.local_path or f"photos://{v.uuid}",
            ))
        return jobs

    def _prepare(self, job: Job) -> None:
        if job.video is not None and job.video.exists():
            return
        self.progress.emit("  downloading from iCloud…")
        path = photos_source.download(job.meta["photos_uuid"], self.cache_dir)
        if self.keep_downloads:
            job.db_path = str(path)
        else:
            job.temp_file = True
        job.video = path
