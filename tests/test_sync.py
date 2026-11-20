import numpy as np
import pytest

from bandkitchen import engine, sync
from tests.synth import SR, make_recording, make_song


@pytest.fixture(scope="module")
def song():
    return make_song(seed=1, bars=64)


@pytest.mark.parametrize("offset_s", [1.974, 0.0, 12.345, 40.0])
def test_recovers_known_offset(song, offset_s):
    audio, events = song
    rec = make_recording(events, len(audio), offset_s, duration_s=60.0)
    r = sync.estimate_offset(rec, audio)
    assert r.offset_s == pytest.approx(offset_s, abs=0.010)


def test_negative_offset_video_starts_before_song(song):
    audio, events = song
    lead_in = 2.5
    rec = make_recording(events, len(audio), -lead_in, duration_s=60.0)
    r = sync.estimate_offset(rec, audio)
    assert r.offset_s == pytest.approx(-lead_in, abs=0.010)


def test_hot_recording_way_over_full_scale(song):
    """Phone audio stored ~17 dB over full scale must still sync (we never clip on input)."""
    audio, events = song
    rec = make_recording(events, len(audio), 5.0, duration_s=60.0, gain=40.0)
    assert np.abs(rec).max() > 4.0
    assert sync.estimate_offset(rec, audio).offset_s == pytest.approx(5.0, abs=0.010)


def test_wrong_song_is_flagged(song):
    audio, _ = song
    _, other_events = make_song(seed=99, bars=64)
    rec = make_recording(other_events, len(audio), 3.0, duration_s=60.0)
    wrong = sync.estimate_offset(rec, audio)
    right_events = song[1]
    right = sync.estimate_offset(make_recording(right_events, len(audio), 3.0, 60.0), audio)
    assert wrong.correlation < right.correlation
    assert not (
        wrong.correlation >= engine.MIN_CORRELATION and wrong.peak_margin >= engine.MIN_PEAK_MARGIN
    )


def test_pure_loop_is_flagged_ambiguous():
    """An identical bar repeated has no unique answer; we must say so, not guess silently."""
    audio, events = make_song(seed=3, bars=48, loop=True)
    rec = make_recording(events, len(audio), 6.0, duration_s=40.0)
    r = sync.estimate_offset(rec, audio)
    assert r.peak_margin < engine.MIN_PEAK_MARGIN


def test_steady_player_shows_no_drift(song):
    audio, events = song
    rec = make_recording(events, len(audio), 4.0, duration_s=90.0)
    r = sync.estimate_offset(rec, audio)
    pts = sync.measure_drift(rec, audio, r.offset_s)
    assert len(pts) >= 5
    assert max(abs(p.deviation_ms) for p in pts) < 10.0


def test_rushing_player_drift_trend(song):
    """A player 0.1% fast drifts ~90 ms early over 90 s; the report should show the trend."""
    audio, events = song
    rec = make_recording(events, len(audio), 4.0, duration_s=90.0, tempo_scale=1.001)
    r = sync.estimate_offset(rec, audio)
    pts = sync.measure_drift(rec, audio, r.offset_s)
    t = np.array([p.video_time_s for p in pts])
    dev = np.array([p.deviation_ms for p in pts])
    slope_ms_per_s = np.polyfit(t, dev, 1)[0]
    assert slope_ms_per_s == pytest.approx(1.0, rel=0.25)  # 0.1% = 1 ms per second


def test_sample_rate_contract():
    assert SR == sync.ANALYSIS_SR
