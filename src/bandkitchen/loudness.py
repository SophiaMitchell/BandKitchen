"""Integrated loudness (LUFS) per ITU-R BS.1770-4 / EBU R128, in numpy/scipy.

K-weighting filter coefficients are derived for any sample rate from the analog prototype
parameters (same approach as libebur128), so we don't need a 48 kHz-only table.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0
BLOCK_S = 0.400
OVERLAP = 0.75


def _shelf(sr: int) -> tuple[np.ndarray, np.ndarray]:
    # Bilinear transform of the analog high-shelf prototype (as in libebur128).
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (g / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = np.array([vh + vb * k / q + k * k, 2.0 * (k * k - vh), vh - vb * k / q + k * k]) / a0
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return b, a


def _highpass(sr: int) -> tuple[np.ndarray, np.ndarray]:
    # RLB high-pass; numerator is exactly [1, -2, 1] per BS.1770 (calibrated with -0.691).
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    return np.array([1.0, -2.0, 1.0]), np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    )


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    b1, a1 = _shelf(sr)
    b2, a2 = _highpass(sr)
    return ss.lfilter(b2, a2, ss.lfilter(b1, a1, x, axis=0), axis=0)


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness of mono (n,) or multichannel (n, ch) float audio.

    Returns -inf for silence. Channels are weighted 1.0 (L/R/C), per BS.1770 for stereo/mono.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = k_weight(x, sr)
    block = int(round(BLOCK_S * sr))
    step = int(round(block * (1 - OVERLAP)))
    if len(y) < block:
        return float("-inf")
    starts = np.arange(0, len(y) - block + 1, step)
    # Per-block mean square per channel, via a cumulative sum.
    csum = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y**2, axis=0)])
    ms = (csum[starts + block] - csum[starts]) / block
    z = ms.sum(axis=1)
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(z)
    gated = z[lk > ABS_GATE_LUFS]
    if len(gated) == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(gated.mean()) + REL_GATE_LU
    with np.errstate(divide="ignore"):
        final = gated[(-0.691 + 10 * np.log10(gated)) > rel]
    return float(-0.691 + 10 * np.log10(final.mean()))
