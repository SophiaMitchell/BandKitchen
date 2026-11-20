"""Play-along splicing: separators, the stem cache, and removal in the render path.

Demucs itself isn't exercised here (it needs PyTorch and a model download); a fake separator
with known stems tests everything around it, and the numpy HPSS fallback runs for real.
"""

import importlib.util

import numpy as np
import pytest
import scipy.signal as ss

from bandkitchen import cache, engine, mix, stems
from tests.synth import SR as SYNTH_SR
from tests.synth import drum_events, render_drums


def _tone(n, sr, f, amp=0.3):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture
def drums_and_pad():
    """Separate 'drums' and 'pad' parts at 44.1 kHz stereo, plus their sum."""
    sr = 44100
    n = sr * 8
    d = render_drums(drum_events(seed=2, bars=4, bpm=120), n, sr, seed=2).astype(np.float32)
    pad = _tone(n, sr, 220) + _tone(n, sr, 330, 0.2)
    d2, p2 = np.stack([d, d], 1), np.stack([pad, pad], 1)
    return d2, p2, d2 + p2


def _energy(x):
    return float((np.asarray(x, dtype=np.float64) ** 2).mean())


class FakeSeparator:
    """Returns pre-baked stems, and counts calls (to check caching)."""

    sample_rate = 44100

    def __init__(self, parts):
        self.parts = parts
        self.calls = 0

    @property
    def cache_id(self):
        return "fake-v1"

    def stems(self):
        return tuple(self.parts)

    def separate(self, audio):
        self.calls += 1
        n = len(audio)
        return {k: v[:n] for k, v in self.parts.items()}


# --- remove_instrument -----------------------------------------------------------------


def test_remove_instrument_sums_the_other_stems():
    a, b, c = (np.full((4, 2), v, np.float32) for v in (1.0, 2.0, 4.0))
    bed = stems.remove_instrument({"drums": a, "bass": b, "other": c}, "drums")
    np.testing.assert_array_equal(bed, np.full((4, 2), 6.0))


def test_remove_instrument_unknown_stem_is_an_error():
    with pytest.raises(ValueError, match="guitar"):
        stems.remove_instrument({"drums": np.zeros((2, 2)), "other": np.zeros((2, 2))}, "guitar")


# --- HPSS fallback ---------------------------------------------------------------------


def test_hpss_stems_sum_back_to_the_input(drums_and_pad):
    *_, mixed = drums_and_pad
    out = stems.HpssSeparator().separate(mixed)
    assert set(out) == {"drums", "other"}
    np.testing.assert_allclose(out["drums"] + out["other"], mixed, atol=1e-4)


def _band_db_removed(original, residual, lo, hi, sr=44100):
    sos = ss.butter(4, [lo, hi], "bp", fs=sr, output="sos")
    e = lambda x: _energy(ss.sosfilt(sos, x[:, 0]))  # noqa: E731
    return 10 * np.log10(e(original) / e(residual))


def test_hpss_removes_snare_and_cymbals_but_not_kick_body(drums_and_pad):
    """Documents the fallback's known behavior; Demucs handles the kick far better."""
    d, p, mixed = drums_and_pad
    bed = stems.remove_instrument(stems.HpssSeparator().separate(mixed), "drums")
    residual = bed - p  # drum sound left in the bed
    assert _band_db_removed(d, residual, 1000, 7000) > 8  # snare
    assert _band_db_removed(d, residual, 7000, 15000) > 8  # hats/cymbals
    assert _band_db_removed(d, residual, 30, 150) < 6  # kick body mostly stays
    assert _energy(bed) > 0.5 * _energy(p)  # the sustained part survives


def test_hpss_accepts_mono():
    out = stems.HpssSeparator().separate(_tone(44100, 44100, 440))
    assert out["drums"].shape == (44100, 2)


# --- separator selection ---------------------------------------------------------------


def test_make_separator_picks_6_stem_model_for_guitar_and_piano():
    assert stems.make_separator("demucs", "drums").stems() == ("drums", "bass", "other", "vocals")
    assert "guitar" in stems.make_separator("demucs", "guitar").stems()
    assert "piano" in stems.make_separator("demucs", "piano").stems()
    with pytest.raises(ValueError):
        stems.make_separator("nope")


@pytest.mark.skipif(importlib.util.find_spec("demucs") is not None, reason="demucs installed")
def test_demucs_missing_gives_install_hint():
    with pytest.raises(stems.SeparatorUnavailable, match=r"\.\[stems\]"):
        stems.DemucsSeparator().separate(np.zeros((44100, 2), np.float32))


# --- cache -----------------------------------------------------------------------------


def test_cache_roundtrip_and_sample_rate_check(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path))
    parts = {"drums": np.ones((10, 2), np.float32), "other": np.zeros((10, 2), np.float32)}
    cache.save_stems("abc123", "fake-v1", 44100, parts)
    got = cache.load_stems("abc123", "fake-v1", 44100)
    assert set(got) == {"drums", "other"}
    np.testing.assert_array_equal(got["drums"], parts["drums"])
    assert cache.load_stems("abc123", "fake-v1", 48000) is None
    assert cache.load_stems("abc123", "other-separator", 44100) is None


def test_incomplete_cache_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path))
    d = tmp_path / "stems" / "abc" / "fake-v1"
    d.mkdir(parents=True)
    np.save(d / "drums.npy", np.ones((3, 2)))  # no meta.json: interrupted write
    assert cache.load_stems("abc", "fake-v1", 44100) is None


def test_file_hash_follows_content_not_name(tmp_path):
    a, b, c = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "c.mp3"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"different")
    assert cache.file_hash(a) == cache.file_hash(b) != cache.file_hash(c)


def test_separate_song_uses_cache_on_second_call(tmp_path, monkeypatch, drums_and_pad):
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(engine.media, "load_audio", lambda path, sr, mono=False: drums_and_pad[2])
    song = tmp_path / "song.mp3"
    song.write_bytes(b"pretend mp3")
    d, p, _ = drums_and_pad
    sep = FakeSeparator({"drums": d, "other": p})
    _, first = engine.separate_song(song, sep)
    _, second = engine.separate_song(song, sep)
    assert (first.cached, second.cached) == (False, True)
    assert sep.calls == 1


# --- mixing a spliced bed --------------------------------------------------------------


def test_bed_keeps_the_full_songs_gain():
    """Removing drums must not push the bed up to the full song's loudness."""
    sr = 48000
    rng = np.random.default_rng(0)
    other = rng.normal(0, 0.1, (sr * 10, 2)).astype(np.float32)
    drums = rng.normal(0, 0.2, (sr * 10, 2)).astype(np.float32)
    full = other + drums
    inst = rng.normal(0, 2.0, (sr * 10, 2)).astype(np.float32)
    _, ref_rep = mix.mix(inst, full, sr)
    _, bed_rep = mix.mix(inst, other, sr, song_reference=full)
    assert bed_rep.song_gain_db == pytest.approx(ref_rep.song_gain_db)
    assert bed_rep.instrument_gain_db == pytest.approx(ref_rep.instrument_gain_db)


def test_spliced_renders_default_to_their_own_level():
    assert mix.default_settings(spliced=False).instrument_rel_db == mix.DEFAULT_INSTRUMENT_DB
    assert mix.default_settings(spliced=True).instrument_rel_db == -2.0


def test_render_rejects_unsupported_instrument_before_any_work(tmp_path):
    sep = FakeSeparator({"drums": np.zeros((4, 2)), "other": np.zeros((4, 2))})
    with pytest.raises(ValueError, match="guitar"):
        engine.render_files(
            "missing.mov", "missing.mp3", tmp_path / "o.mp4", remove="guitar", separator=sep
        )
    assert sep.calls == 0


def test_synth_rate_matches_sync_rate():
    from bandkitchen import sync

    assert SYNTH_SR == sync.ANALYSIS_SR
