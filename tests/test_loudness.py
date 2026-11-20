import numpy as np
import pytest

from bandkitchen.loudness import integrated_lufs


def sine(sr, amp=1.0, f=997.0, secs=5.0):
    t = np.arange(int(secs * sr)) / sr
    return amp * np.sin(2 * np.pi * f * t)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_reference_tone_mono(sr):
    # BS.1770: a 997 Hz full-scale sine in one channel reads -3.01 LUFS.
    assert integrated_lufs(sine(sr), sr) == pytest.approx(-3.01, abs=0.05)


def test_stereo_sums_channels():
    x = sine(48000)
    assert integrated_lufs(np.stack([x, x], axis=1), 48000) == pytest.approx(0.0, abs=0.05)


def test_gain_maps_one_to_one():
    assert integrated_lufs(sine(48000, amp=0.1), 48000) == pytest.approx(-23.01, abs=0.05)


def test_over_full_scale_is_measured_not_clipped():
    assert integrated_lufs(sine(48000, amp=10.0), 48000) == pytest.approx(16.99, abs=0.05)


def test_silence_and_short_input():
    assert integrated_lufs(np.zeros(48000 * 2), 48000) == float("-inf")
    assert integrated_lufs(np.ones(100), 48000) == float("-inf")


def test_relative_gate_ignores_quiet_passages():
    sr = 48000
    loud, quiet = sine(sr, 0.5, secs=10), sine(sr, 0.5 * 10 ** (-30 / 20), secs=10)
    assert integrated_lufs(np.concatenate([loud, quiet]), sr) == pytest.approx(
        integrated_lufs(loud, sr), abs=0.2
    )
