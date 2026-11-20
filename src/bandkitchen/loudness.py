"""Integrated loudness (LUFS) per ITU-R BS.1770-4 / EBU R128, in numpy/scipy.

K-weighting filter coefficients are derived for any sample rate from the analog prototype
parameters (same approach as libebur128), so we don't need a 48 kHz-only table.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

# K-weighting gain at the 997 Hz reference tone, subtracted so that tone reads at its
# unweighted level (see integrated_lufs). Defined by BS.1770.
K_WEIGHT_OFFSET_DB = -0.691
ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0
BLOCK_S = 0.400
OVERLAP = 0.75


def _shelf(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1 of K-weighting: a high shelf that models the acoustic effect of the head
    (sound above ~2 kHz is perceived about 4 dB louder).

    BS.1770 publishes this filter only as biquad coefficients at 48 kHz (Annex 1, Table 1).
    The constants below are the analog prototype parameters that reproduce those
    coefficients exactly (the derivation libebur128 and ffmpeg use). Re-applying the bilinear
    transform with them gives the same filter at any sample rate. They are part of the
    standard, not tuned to any audio, so don't round or "improve" them.
    """
    # f0: shelf corner frequency, Hz (~1682 Hz)
    # g:  high-frequency boost, dB (~+4 dB)
    # q:  shelf quality factor (~0.707, i.e. 1/sqrt(2): a smooth, Butterworth-like transition)
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)  # pre-warped analog frequency for the bilinear transform
    # vh: the shelf's high-frequency gain as a linear amplitude ratio (+4 dB -> ~1.585x)
    vh = 10 ** (g / 20.0)
    # vb: gain at the transition band, about sqrt(vh) (the geometric midpoint of 1x and vh).
    # The exponent is fractionally under 0.5 so the result matches the published 48 kHz
    # coefficients to full precision.
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = np.array([vh + vb * k / q + k * k, 2.0 * (k * k - vh), vh - vb * k / q + k * k]) / a0
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return b, a


def _highpass(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2 of K-weighting: the "RLB" (revised low-frequency B-curve) high-pass, which
    discounts deep bass that people perceive as less loud than its energy suggests.

    BS.1770 publishes it at 48 kHz (Annex 1, Table 2). As with the shelf, the constants are
    the analog parameters that reproduce that table, so it works at any sample rate. The
    numerator stays exactly [1, -2, 1] as in the standard: the K_WEIGHT_OFFSET_DB calibration
    was computed with that numerator, so normalizing it would shift
    every reading slightly.
    """
    # f0: cutoff frequency, Hz (~38 Hz; content below this is progressively ignored)
    # q:  quality factor (~0.5: a gentle, critically damped roll-off with no resonant bump)
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


def _lufs(mean_square: np.ndarray | float) -> np.ndarray | float:
    """Mean square (summed over channels) -> loudness in LUFS."""
    return K_WEIGHT_OFFSET_DB + 10 * np.log10(mean_square)


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness of mono (n,) or multichannel (n, ch) float audio.

    Returns -inf for silence. Channels are weighted 1.0 (L/R/C), per BS.1770 for stereo/mono.

    Steps: K-weight each channel, take the mean square in 400 ms blocks overlapping by 75%,
    sum across channels, drop blocks below -70 LUFS (absolute gate), then drop blocks more
    than 10 LU below the average of what's left (relative gate), and average the rest.

    Where K_WEIGHT_OFFSET_DB (-0.691) comes from: loudness is
    ``-0.691 + 10*log10(mean square)`` (see ``_lufs``). K-weighting is
    not flat at 1 kHz: at 997 Hz (the standard's reference tone) the two filters together
    boost the signal by about +0.691 dB. Subtracting 0.691 dB cancels that, so a 997 Hz tone
    reads the same as its unweighted level. A full-scale sine has a mean square of 0.5, so
    in one channel it reads exactly 10*log10(0.5) = -3.01 LUFS (checked in the tests). The
    constant is defined by BS.1770 itself, which is why every compliant meter agrees.
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
        lk = _lufs(z)
    gated = z[lk > ABS_GATE_LUFS]
    if len(gated) == 0:
        return float("-inf")
    rel = _lufs(gated.mean()) + REL_GATE_LU
    with np.errstate(divide="ignore"):
        final = gated[_lufs(gated) > rel]
    return float(_lufs(final.mean()))
