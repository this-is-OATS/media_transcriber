#!/usr/bin/env python3
"""The queue ORDER, on real files made for the test.

Johnny, 2026-11-02: "add a new file and watch it interleave". The nightly's own
--dry-run could not answer that — pending was 0, so there was no order to look
at. This makes new files and reads the order off the real functions.

Why a test and not a one-off script: `pending=0` is going to be the normal
state from now on. The backlog is transcribed, so the interleave only ever
governs files that arrive later — a handful at a time, never enough on their
own to show a 1-in-4 pattern. Twelve files made on purpose show it in one line.

    python3 tests/test_order.py

Runs with the system python: `whisper` is stubbed, because app/transcriber.py
imports it at module scope and nothing here transcribes anything.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("whisper", types.ModuleType("whisper"))

import batch as B  # noqa: E402

ok: list[str] = []


def t(name: str, cond: bool) -> None:
    assert cond, name
    ok.append(name)


def kinds(queue: list[Path]) -> str:
    """The order as one readable string: F footage, S screen recording."""
    return "".join("S" if str(f) in B.SCREEN_PATHS else "F" for f in queue)


# --- the pure function, first: nothing is dropped, whatever the two sizes ---
t("one in four, exactly",
  B.interleave(list("abcdefghi"), list("XYZ"), 4) == list("abcXdefYghiZ"))
t("no screen recordings -> the main order, untouched",
  B.interleave(list("abc"), [], 4) == list("abc"))
t("no footage -> every screen recording, in order",
  B.interleave([], list("XYZ"), 4) == list("XYZ"))
t("footage runs dry first -> the rest of the screens follow, none dropped",
  B.interleave(list("abc"), list("XYZ"), 4) == list("abcXYZ"))
t("screens run dry first -> the rest of the footage follows, none dropped",
  B.interleave(list("abcdefgh"), ["X"], 4) == list("abcXdefgh"))
for m, s in ((9, 3), (3, 9), (100, 7), (1, 1), (0, 5), (5, 0), (40, 13)):
    out = B.interleave([f"m{i}" for i in range(m)], [f"s{i}" for i in range(s)], 4)
    t(f"nothing dropped and nothing duplicated at main={m} screen={s}",
      len(out) == m + s and len(set(out)) == m + s)

# --- and now on files that did not exist a moment ago ------------------------
with tempfile.TemporaryDirectory(prefix="order-probe-") as tmp:
    root = Path(tmp)
    # Distinct sizes so smallest-first is deterministic; the names are the ones
    # the capture tools really write.
    made: list[Path] = []
    for i in range(1, 10):
        p = root / f"clip-{i:02d}.mov"
        p.write_bytes(b"\0" * (i * 2048))
        made.append(p)
    for i in range(1, 4):
        p = root / f"Screen Recording 2026-11-02 at 10.0{i}.mov"
        p.write_bytes(b"\0" * (i * 1024))
        made.append(p)

    B.SCREEN_PATHS.clear()
    files = B.collect([root])
    t("every new file was found", len(files) == 12)
    t("the three screen recordings were recognised by name",
      len(B.SCREEN_PATHS) == 3)

    order = B.order_queue(files, "interleave")
    t("nothing was dropped from the queue", len(order) == 12)
    t(f"the order is FFFS FFFS FFFS, got {kinds(order)}",
      kinds(order) == "FFFSFFFSFFFS")
    t("within each group, still smallest first",
      [f.stat().st_size for f in order if str(f) not in B.SCREEN_PATHS]
      == sorted(f.stat().st_size for f in order if str(f) not in B.SCREEN_PATHS))

    # A SINGLE new screen recording arriving - the literal ask. It must not be
    # parked at the end of the footage; slot 4 is where it belongs.
    B.SCREEN_PATHS.clear()
    one = root / "Screen Recording 2026-11-02 at 11.00.mov"
    one.write_bytes(b"\0" * 512)
    files = B.collect([root])
    order = B.order_queue([f for f in files if "at 10.0" not in f.name], "interleave")
    t(f"one new screen recording lands in slot 4, got {kinds(order)}",
      kinds(order) == "FFFSFFFFFF")

    # --- the seam the nightly actually uses: the manifest's own column -------
    # Here NO filename says "screen". Only the CSV does, which is how the real
    # run classifies a Photos row whose name is an opaque UUID.
    B.SCREEN_PATHS.clear()
    B.DURATIONS.clear()
    man = root / "probe_manifest.csv"
    cols = ["source", "bucket", "path", "filename", "size_bytes", "duration_s",
            "has_audio", "dupe_of", "photos_uuid", "screen_recording"]
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(1, 13):
            w.writerow({
                "source": "nas", "bucket": "ORDER PROBE",
                "path": f"/Volumes/NAS/ORDER PROBE/row-{i:02d}.mov",
                "filename": f"row-{i:02d}.mov",
                "size_bytes": i * 2048, "duration_s": 60 + i,
                "has_audio": "True", "dupe_of": "", "photos_uuid": "",
                # every fourth row is a screen recording BY THE COLUMN ONLY
                "screen_recording": "True" if i % 4 == 0 else "False",
            })
    rows = B.manifest_files(man)
    t("every manifest row was queued", len(rows) == 12)
    t("the column alone marked three rows, with no help from the filename",
      len(B.SCREEN_PATHS) == 3
      and not any(B.is_screen_name(p.name) for p in rows))
    order = B.order_queue(rows, "interleave")
    t(f"the manifest seam interleaves the same way, got {kinds(order)}",
      kinds(order) == "FFFSFFFSFFFS")

    # --- the policies that stayed, by name ---------------------------------
    t("skip leaves them out", len(B.order_queue(rows, "skip")) == 9)
    t("last puts all three behind the footage",
      kinds(B.order_queue(rows, "last")) == "FFFFFFFFFSSS")
    t("inline does not reorder at all",
      B.order_queue(rows, "inline") == rows)

print("\n".join("  ok  " + s for s in ok))
print(f"\n{len(ok)} assertions passed")
