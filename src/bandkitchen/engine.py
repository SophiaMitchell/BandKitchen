"""File-level operations shared by the CLI and (later) the HTTP API."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal as ss

from . import cache, media, mix, render, stems, sync

# Below these, the match is suspect: likely a different song or recording, or too little overlap.
MIN_CORRELATION = 0.25
MIN_PEAK_MARGIN = 0.20


@dataclass
class SyncReport:
    video: str
    song: str
    result: sync.SyncResult
    reliable: bool
    drift: list[sync.DriftPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "song": self.song,
            **self.result.to_dict(),
            "reliable": self.reliable,
            "drift": [p.to_dict() for p in self.drift],
        }


def sync_files(video: str | Path, song: str | Path, drift_report: bool = False) -> SyncReport:
    v = media.load_audio(video, sync.ANALYSIS_SR, mono=True)
    s = media.load_audio(song, sync.ANALYSIS_SR, mono=True)
    result = sync.estimate_offset(v, s)
    reliable = result.correlation >= MIN_CORRELATION and result.peak_margin >= MIN_PEAK_MARGIN
    drift = sync.measure_drift(v, s, result.offset_s) if drift_report else []
    return SyncReport(str(video), str(song), result, reliable, drift)


@dataclass
class StemsReport:
    song: str
    separator: str
    stems: list[str]
    cached: bool
    cache_dir: str | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def separate_song(
    song: str | Path, separator: stems.Separator, use_cache: bool = True
) -> tuple[dict[str, np.ndarray], StemsReport]:
    """Split ``song`` into stems at ``separator.sample_rate``, using the cache when possible."""
    key = cache.file_hash(song) if use_cache else None
    if key:
        hit = cache.load_stems(key, separator.cache_id, separator.sample_rate)
        if hit is not None:
            return hit, StemsReport(str(song), separator.cache_id, sorted(hit), True, None)
    audio = media.load_audio(song, separator.sample_rate)
    out = separator.separate(audio)
    where = None
    if key:
        where = str(cache.save_stems(key, separator.cache_id, separator.sample_rate, out))
    return out, StemsReport(str(song), separator.cache_id, sorted(out), False, where)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return ss.resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


@dataclass
class RenderReport:
    output: str
    offset_s: float
    settings: mix.MixSettings
    mix: mix.MixReport
    sync: SyncReport | None
    removed: str | None = None
    stems: StemsReport | None = None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "offset_s": self.offset_s,
            "settings": self.settings.to_dict(),
            "mix": self.mix.to_dict(),
            "sync": self.sync.to_dict() if self.sync else None,
            "removed": self.removed,
            "stems": self.stems.to_dict() if self.stems else None,
        }


def render_files(
    video: str | Path,
    song: str | Path,
    out: str | Path,
    settings: mix.MixSettings | None = None,
    offset_s: float | None = None,
    remove: str | None = None,
    separator: stems.Separator | None = None,
    use_cache: bool = True,
) -> RenderReport:
    """Sync (unless ``offset_s`` is given), mix, and write ``out``.

    ``remove``: an instrument to take out of the song first (play-along splicing). Sync
    still uses the full song: its drums are exactly what a drum take should line up with.
    """
    sep = None
    if remove:  # check before any slow work
        sep = separator or stems.make_separator("demucs", remove)
        if remove not in sep.stems():
            raise ValueError(f"{sep.cache_id} can't isolate {remove!r}; it provides {sep.stems()}")
    sync_report = None
    if offset_s is None:
        sync_report = sync_files(video, song)
        offset_s = sync_report.result.offset_s
    sr = mix.RENDER_SR
    inst = mix.to_stereo(media.load_audio(video, sr))
    full = mix.align_song(media.load_audio(song, sr), offset_s, len(inst), sr)
    settings = settings or mix.MixSettings()

    stems_report = None
    if sep is not None:
        parts, stems_report = separate_song(song, sep, use_cache=use_cache)
        bed = _resample(stems.remove_instrument(parts, remove), sep.sample_rate, sr)
        bed = mix.align_song(bed, offset_s, len(inst), sr)
        mixed, mix_report = mix.mix(inst, bed, sr, settings, song_reference=full)
    else:
        mixed, mix_report = mix.mix(inst, full, sr, settings)
    render.mux(video, mixed, sr, out)
    return RenderReport(str(out), offset_s, settings, mix_report, sync_report, remove, stems_report)
