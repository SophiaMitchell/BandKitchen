"""Find where a play-along video sits in the original song.

Convention: ``offset_s`` is the song time at video t=0, i.e. ``song_time = video_time + offset_s``.
Positive means the video starts partway into the song (the usual case); negative means the
video starts before the song does.

Method: multi-band onset-strength envelopes (spectral flux in 5 bands, ~12 ms frames) are
cross-correlated over the whole overlap with GCC-PHAT weighting, and the bands summed.
- Separate bands keep a kick hit from matching a snare hit.
- PHAT (spectral whitening, partial with PHAT_BETA) de-emphasizes the strongly periodic
  part of the envelopes (steady hi-hats, four-on-the-floor), which otherwise makes a
  match one bar or one section away score almost as well as the true one.
- Matching the whole overlap (not short windows) uses intros, breaks and fills to pin down
  the unique alignment. Parabolic interpolation gives a sub-frame estimate.

Accuracy note: two drummers never hit at identical times. On real recordings the kick and
snare bands can disagree by 10-20 ms, so "the" offset is only defined to about that; the UI
offers a nudge for the final touch.

Drift correction is deliberately NOT applied anywhere: a single global offset keeps the
player's real timing visible. ``measure_drift`` reports local timing as a practice diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.signal as ss

ANALYSIS_SR = 22050
HOP = 256
N_FFT = 2048
FRAME_S = HOP / ANALYSIS_SR
BAND_EDGES_HZ = (0.0, 150.0, 400.0, 2000.0, 6000.0, ANALYSIS_SR / 2 + 1)
EXCLUSION_S = 0.25  # around the best peak when looking for the runner-up
PHAT_BETA = 0.75  # 0 = plain cross-correlation, 1 = full PHAT whitening


@dataclass(frozen=True)
class SyncResult:
    offset_s: float
    correlation: float  # normalized envelope correlation at offset_s, over the overlap
    peak_margin: float  # 1 - runner_up/best (whitened), runner-up >= EXCLUSION_S away;
    # low = ambiguous (wrong song, too little overlap, or a pure loop)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DriftPoint:
    video_time_s: float  # centre of the analysis window, in video time
    deviation_ms: float  # local offset - global offset; positive = player early (rushing)
    strength: float  # normalized local correlation; low = little to match (e.g. a break)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def _standardize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-12)


def onset_bands(x: np.ndarray) -> np.ndarray:
    """Per-band log spectral flux, detrended and standardized. Shape (n_bands, n_frames).

    Expects audio at ANALYSIS_SR.
    """
    x = _as_mono(x)
    f, _, z = ss.stft(
        x, ANALYSIS_SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False
    )
    s = np.log1p(100.0 * np.abs(z)).astype(np.float32)
    out = []
    for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:], strict=True):
        flux = np.maximum(0.0, np.diff(s[(f >= lo) & (f < hi)], axis=1)).sum(axis=0)
        flux = np.maximum(flux - ss.medfilt(flux, 31), 0.0)
        out.append(_standardize(flux))
    return np.vstack(out)


def _band_xcorr(video: np.ndarray, song: np.ndarray, mode: str = "full") -> np.ndarray:
    return sum(
        ss.correlate(s, v, mode=mode, method="fft") for v, s in zip(video, song, strict=True)
    )


def _band_gcc_phat(video: np.ndarray, song: np.ndarray, beta: float = PHAT_BETA) -> np.ndarray:
    """Whitened full cross-correlation, summed over bands. Index 0 = lag -(n_video - 1)."""
    nv, ns = video.shape[1], song.shape[1]
    n_fft = 1 << (nv + ns - 2).bit_length()
    total = np.zeros(nv + ns - 1)
    for v, s in zip(video, song, strict=True):
        g = np.fft.rfft(s, n_fft) * np.conj(np.fft.rfft(v, n_fft))
        g /= (np.abs(g) + 1e-9) ** beta
        c = np.fft.irfft(g, n_fft)
        total += np.concatenate([c[n_fft - (nv - 1) :], c[:ns]])
    return total


def _parabolic(c: np.ndarray, i: int) -> float:
    if 0 < i < len(c) - 1:
        a, b, d = c[i - 1], c[i], c[i + 1]
        denom = a - 2 * b + d
        if denom != 0:
            return i + 0.5 * (a - d) / denom
    return float(i)


def _norm(video: np.ndarray, song: np.ndarray, peak: float) -> float:
    return float(peak / np.sqrt((video**2).sum() * (song**2).sum()))


def estimate_offset(video_audio: np.ndarray, song_audio: np.ndarray) -> SyncResult:
    """Estimate ``offset_s`` (song time at video t=0). Inputs: float audio at ANALYSIS_SR."""
    bv, bs = onset_bands(video_audio), onset_bands(song_audio)
    c = _band_gcc_phat(bv, bs)
    lag0 = -(bv.shape[1] - 1)  # lag (song frame - video frame) of c[0]
    i = int(np.argmax(c))

    ex = int(round(EXCLUSION_S / FRAME_S))
    masked = c.copy()
    masked[max(0, i - ex) : i + ex + 1] = -np.inf
    runner_up = max(float(masked.max()), 0.0)
    margin = 0.0 if c[i] <= 0 else float(np.clip(1.0 - runner_up / c[i], 0.0, 1.0))

    return SyncResult(
        offset_s=float((lag0 + _parabolic(c, i)) * FRAME_S),
        correlation=_overlap_correlation(bv, bs, lag0 + i),
        peak_margin=margin,
    )


def _overlap_correlation(bv: np.ndarray, bs: np.ndarray, lag: int) -> float:
    """Pearson-style correlation of the band envelopes where video and song overlap."""
    v0, s0 = max(0, -lag), max(0, lag)
    n = min(bv.shape[1] - v0, bs.shape[1] - s0)
    if n <= 1:
        return 0.0
    v, s = bv[:, v0 : v0 + n], bs[:, s0 : s0 + n]
    return _norm(v, s, float((v * s).sum()))


def measure_drift(
    video_audio: np.ndarray,
    song_audio: np.ndarray,
    offset_s: float,
    window_s: float = 16.0,
    hop_s: float = 8.0,
    search_s: float = 0.15,
) -> list[DriftPoint]:
    """Local timing through the video, searched only within +/- ``search_s`` of the global
    offset (so it can't jump a whole beat). A practice diagnostic; never applied to the mix.
    """
    bv, bs = onset_bands(video_audio), onset_bands(song_audio)
    n_win = int(round(window_s / FRAME_S))
    n_hop = int(round(hop_s / FRAME_S))
    off = offset_s / FRAME_S
    pad = int(np.ceil(search_s / FRAME_S))
    points: list[DriftPoint] = []
    for start in range(0, max(bv.shape[1] - n_win, 0) + 1, n_hop):
        seg = bv[:, start : start + n_win]
        lo = int(np.floor(start + off)) - pad
        hi = int(np.ceil(start + n_win + off)) + pad
        if lo < 0 or hi > bs.shape[1]:
            continue
        ref = bs[:, lo:hi]
        c = _band_xcorr(seg, ref, mode="valid")
        j = int(np.argmax(c))
        local_s = (lo + _parabolic(c, j) - start) * FRAME_S
        points.append(
            DriftPoint(
                video_time_s=(start + n_win / 2) * FRAME_S,
                deviation_ms=(local_s - offset_s) * 1000.0,
                strength=_norm(seg, ref[:, j : j + n_win], float(c[j])),
            )
        )
    return points
