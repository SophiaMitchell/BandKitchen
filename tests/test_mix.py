import numpy as np
import pytest

from bandkitchen import mix
from bandkitchen.loudness import integrated_lufs

SR = 48000


def noise(secs, rms, seed=0):
    return np.random.default_rng(seed).normal(0, rms, (int(secs * SR), 2)).astype(np.float32)


def test_align_song_positive_offset_trims_song_start():
    song = np.arange(10, dtype=np.float32)
    out = mix.align_song(song, offset_s=3 / SR, n_out=5, sr=SR)
    assert out[:, 0].tolist() == [3, 4, 5, 6, 7]


def test_align_song_negative_offset_pads_silence():
    song = np.arange(1, 11, dtype=np.float32)
    out = mix.align_song(song, offset_s=-2 / SR, n_out=5, sr=SR)
    assert out[:, 0].tolist() == [0, 0, 1, 2, 3]


def test_align_song_shorter_than_video_pads_end():
    out = mix.align_song(np.ones(3, dtype=np.float32), 0.0, n_out=6, sr=SR)
    assert out[:, 1].tolist() == [1, 1, 1, 0, 0, 0]


def test_limiter_enforces_ceiling_exactly():
    x = noise(2, 0.3)
    x[SR // 2] = 6.0
    x[SR] = -9.0
    y, red = mix.limit(x, SR, ceiling_dbfs=-1.0)
    assert np.abs(y).max() <= mix.db_to_gain(-1.0) + 1e-6
    assert red > 15


def test_limiter_leaves_quiet_audio_untouched():
    x = noise(1, 0.05)
    y, red = mix.limit(x, SR)
    assert red == 0.0
    np.testing.assert_array_equal(x, y)


def test_limiter_ramps_rather_than_steps():
    x = np.full((SR, 2), 0.5, dtype=np.float32)
    x[SR // 2] = 4.0
    y, _ = mix.limit(x, SR, window_ms=20)
    g = y[:, 0] / x[:, 0]
    assert np.max(np.abs(np.diff(g))) < 0.1  # no hard gain jump


def test_mix_hits_targets_for_hot_phone_audio():
    """The phone track is ~+5 LUFS (far over full scale); the mix must normalize it."""
    inst = noise(20, 4.0, seed=1)  # way over full scale
    song = noise(20, 0.3, seed=2)
    settings = mix.MixSettings(song_lufs=-14.0, instrument_rel_db=-3.0)
    out, rep = mix.mix(inst, song, SR, settings)
    assert rep.instrument_lufs_in > 0
    assert rep.song_lufs_in + rep.song_gain_db == pytest.approx(-14.0, abs=0.01)
    assert rep.instrument_lufs_in + rep.instrument_gain_db == pytest.approx(-17.0, abs=0.01)
    assert np.abs(out).max() <= mix.db_to_gain(settings.ceiling_dbfs) + 1e-6
    assert rep.output_peak_dbfs <= settings.ceiling_dbfs + 1e-4


def test_instrument_level_is_relative_to_song():
    inst, song = noise(20, 1.0, 3), noise(20, 0.2, 4)
    _, a = mix.mix(inst, song, SR, mix.MixSettings(instrument_rel_db=0.0))
    _, b = mix.mix(inst, song, SR, mix.MixSettings(instrument_rel_db=-6.0))
    assert a.instrument_gain_db - b.instrument_gain_db == pytest.approx(6.0, abs=0.01)


def test_mix_silent_instrument_does_not_explode():
    inst = np.zeros((SR * 5, 2), dtype=np.float32)
    out, rep = mix.mix(inst, noise(5, 0.2), SR)
    assert np.isfinite(out).all()
    assert rep.instrument_gain_db == 0.0
    assert integrated_lufs(out, SR) == pytest.approx(-14.0, abs=0.1)


def test_mix_rejects_unaligned_lengths():
    with pytest.raises(ValueError):
        mix.mix(noise(1, 0.1), noise(2, 0.1), SR)
