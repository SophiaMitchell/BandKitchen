"""Gain staging and mixing, all in float. Nothing is clipped until the final limiter.

Levels are set by loudness (LUFS), not raw percentages:
- the song is normalized to ``song_lufs``;
- the player's (camera) audio is normalized to ``song_lufs + instrument_rel_db``.
So a preset like "drums -3 dB under the song" transfers to any recording, however hot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from .loudness import integrated_lufs

RENDER_SR = 48000


@dataclass(frozen=True)
class MixSettings:
    song_lufs: float = -14.0  # streaming-ish target for the song bed
    instrument_rel_db: float = -3.0  # player's audio relative to the song
    ceiling_dbfs: float = -1.0  # limiter sample-peak ceiling
    limiter_window_ms: float = 20.0  # gain ramps down over this window ahead of a peak

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MixReport:
    song_lufs_in: float
    instrument_lufs_in: float
    song_gain_db: float
    instrument_gain_db: float
    limiter_max_reduction_db: float
    output_lufs: float
    output_peak_dbfs: float

    def to_dict(self) -> dict:
        return asdict(self)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def to_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


def align_song(song: np.ndarray, offset_s: float, n_out: int, sr: int) -> np.ndarray:
    """Place the song on the video timeline: out[t] = song[t + offset_s], zero-padded."""
    song = to_stereo(song)
    shift = int(round(offset_s * sr))
    out = np.zeros((n_out, 2), dtype=np.float32)
    src_start = max(shift, 0)
    dst_start = max(-shift, 0)
    n = min(n_out - dst_start, len(song) - src_start)
    if n > 0:
        out[dst_start : dst_start + n] = song[src_start : src_start + n]
    return out


def limit(
    x: np.ndarray, sr: int, ceiling_dbfs: float = -1.0, window_ms: float = 20.0
) -> tuple[np.ndarray, float]:
    """Look-ahead peak limiter. Guarantees |y| <= ceiling at every sample.

    Gain = moving average (length L) of a running minimum (length L) of the required gain.
    Every averaged value is <= the required gain at each peak, so the ceiling holds exactly,
    while the gain glides down over L samples before the peak and back up after it.
    Returns (y, max_reduction_db).
    """
    x = to_stereo(x)
    ceiling = db_to_gain(ceiling_dbfs)
    peak = np.abs(x).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if need.min() >= 1.0:
        return x.copy(), 0.0
    n = max(int(round(window_ms * 1e-3 * sr)), 1)
    # trailing min: m[k] = min(need[k-n+1 .. k])
    m = minimum_filter1d(need, size=n, mode="nearest", origin=(n - 1) // 2)
    # leading mean: g[p] = mean(m[p .. p+n-1])
    g = uniform_filter1d(m, size=n, mode="nearest", origin=-(n // 2))
    g = np.minimum(g, need)  # guard against float rounding at the edges
    y = (x * g[:, None]).astype(np.float32)
    return y, float(-20 * np.log10(g.min()))


def mix(
    instrument: np.ndarray, song_aligned: np.ndarray, sr: int, settings: MixSettings | None = None
) -> tuple[np.ndarray, MixReport]:
    """Mix the player's audio with the (already aligned) song. Both float, same length."""
    s = settings or MixSettings()
    inst, song = to_stereo(instrument), to_stereo(song_aligned)
    if len(inst) != len(song):
        raise ValueError("instrument and song must be the same length; align the song first")
    song_in = integrated_lufs(song, sr)
    inst_in = integrated_lufs(inst, sr)
    song_g = s.song_lufs - song_in if np.isfinite(song_in) else 0.0
    inst_g = (s.song_lufs + s.instrument_rel_db) - inst_in if np.isfinite(inst_in) else 0.0
    summed = inst * db_to_gain(inst_g) + song * db_to_gain(song_g)
    out, reduction = limit(summed, sr, s.ceiling_dbfs, s.limiter_window_ms)
    report = MixReport(
        song_lufs_in=song_in,
        instrument_lufs_in=inst_in,
        song_gain_db=float(song_g),
        instrument_gain_db=float(inst_g),
        limiter_max_reduction_db=reduction,
        output_lufs=integrated_lufs(out, sr),
        output_peak_dbfs=float(20 * np.log10(max(np.abs(out).max(), 1e-12))),
    )
    return out, report
