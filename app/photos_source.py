"""Apple Photos library as a transcription source (read-only).

Uses osxphotos to list videos. Most libraries keep originals in iCloud, so
videos that aren't on disk are downloaded one at a time into a cache folder,
transcribed, and then deleted (unless `keep_downloads` is set).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logging.getLogger("osxphotos").setLevel(logging.ERROR)

_photosdb = None


def _db():
    """PhotosDB takes several seconds to load, so keep one per process."""
    global _photosdb
    if _photosdb is None:
        import osxphotos  # heavy import; only when scanning
        # Pass the library path explicitly: without it osxphotos reads Photos'
        # sandboxed prefs plist, which macOS blocks for app-launched Python.
        lib = os.environ.get("PHOTOS_LIBRARY") or str(
            Path.home() / "Pictures" / "Photos Library.photoslibrary")
        _photosdb = osxphotos.PhotosDB(dbfile=lib)
    return _photosdb


@dataclass
class PhotoVideo:
    uuid: str
    original_filename: str
    taken_at: str            # ISO timestamp
    location: str | None
    albums: list[str] = field(default_factory=list)
    local_path: str | None = None   # None = only in iCloud

    @property
    def output_name(self) -> str:
        """e.g. '2023-04-23 IMG_0766' — date first so transcripts sort by day."""
        stem = Path(self.original_filename).stem
        name = f"{self.taken_at[:10]} {stem}"
        return re.sub(r'[/:\\]', "-", name)

    def meta(self) -> dict:
        return {
            "source": "photos",
            "photos_uuid": self.uuid,
            "taken_at": self.taken_at,
            "location": self.location,
            "albums": ", ".join(self.albums) or None,
            "original_filename": self.original_filename,
        }


def list_videos(include_shared: bool = False) -> list[PhotoVideo]:
    db = _db()
    out: list[PhotoVideo] = []
    for p in db.photos(images=False, movies=True):
        if p.shared and not include_shared:
            continue
        if p.intrash:
            continue
        out.append(PhotoVideo(
            uuid=p.uuid,
            original_filename=p.original_filename or p.filename,
            taken_at=p.date.isoformat(),
            location=p.place.name if p.place else None,
            albums=list(p.albums),
            local_path=p.path,
        ))
    out.sort(key=lambda v: v.taken_at)
    return out


def download(uuid: str, dest_dir: Path) -> Path:
    """Fetch the original video from iCloud into dest_dir. Returns its path."""
    from osxphotos.photoexporter import ExportOptions, PhotoExporter

    # One folder per video: original filenames (IMG_0001.MOV) repeat across years.
    dest_dir = dest_dir / uuid
    dest_dir.mkdir(parents=True, exist_ok=True)
    photo = _db().get_photo(uuid)
    if photo is None:
        raise RuntimeError(f"photo {uuid} not found in library")

    errors = []
    # PhotoKit first (fast, needs Photos permission); AppleScript as fallback
    # (drives Photos.app, needs Automation permission).
    for opts in (
        ExportOptions(download_missing=True, use_photokit=True, overwrite=True),
        ExportOptions(download_missing=True, use_photos_export=True, overwrite=True),
    ):
        try:
            res = PhotoExporter(photo).export(str(dest_dir), options=opts)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue
        for f in res.exported:
            if Path(f).suffix.lower() in {".mov", ".mp4", ".m4v"}:
                return Path(f)
        errors.append("export produced no video file")

    hint = ""
    if any("auth" in e.lower() for e in errors):
        hint = (" — allow Photos access: System Settings > Privacy & Security > "
                "Photos > Media Transcriber")
    raise RuntimeError(f"iCloud download failed: {errors[-1]}{hint}")
