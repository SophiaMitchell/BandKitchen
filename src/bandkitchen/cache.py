"""On-disk cache for expensive per-song analysis (currently: separated stems).

Keyed by a hash of the song file's bytes, so renaming or moving a file keeps its cache, and
editing it invalidates the cache. Location: $BANDKITCHEN_CACHE, else ~/.bandkitchen-cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def cache_root() -> Path:
    return Path(os.environ.get("BANDKITCHEN_CACHE", Path.home() / ".bandkitchen-cache"))


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _stem_dir(song_hash: str, separator_id: str) -> Path:
    return cache_root() / "stems" / song_hash[:32] / separator_id


def load_stems(song_hash: str, separator_id: str, sample_rate: int) -> dict[str, np.ndarray] | None:
    d = _stem_dir(song_hash, separator_id)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("sample_rate") != sample_rate:
        return None
    try:
        return {name: np.load(d / f"{name}.npy") for name in meta["stems"]}
    except (OSError, ValueError):
        return None  # partial or corrupt cache: recompute


def save_stems(
    song_hash: str, separator_id: str, sample_rate: int, stems: dict[str, np.ndarray]
) -> Path:
    d = _stem_dir(song_hash, separator_id)
    d.mkdir(parents=True, exist_ok=True)
    for name, x in stems.items():
        np.save(d / f"{name}.npy", np.asarray(x, dtype=np.float32))
    # meta.json is written last, so its presence means the stems are complete.
    (d / "meta.json").write_text(json.dumps({"sample_rate": sample_rate, "stems": sorted(stems)}))
    return d
