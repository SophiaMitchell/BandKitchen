"""Synthetic 'songs' and 'play-along recordings' with known offsets, for tests.

The song has sections (verse groove, chorus groove, fills every 4th bar) so that, like real
music, it is mostly but not perfectly periodic. The 'recording' re-synthesizes only the drum
part with different timbre and room sound, drives it far above full scale (like a phone mic
next to a kit), adds noise, and crops it to start ``offset_s`` into the song.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

SR = 22050


def _kick(sr, decay=0.12, f0=55.0):
    t = np.arange(int(0.35 * sr)) / sr
    f = f0 + 90 * np.exp(-t / 0.02)
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / decay)


def _snare(sr, rng, decay=0.10, tone=190.0):
    t = np.arange(int(0.25 * sr)) / sr
    noise = ss.sosfilt(
        ss.butter(2, [1000, 7000], "bp", fs=sr, output="sos"), rng.normal(size=len(t))
    )
    return (0.6 * noise + 0.4 * np.sin(2 * np.pi * tone * t)) * np.exp(-t / decay)


def _hat(sr, rng, decay=0.03):
    t = np.arange(int(0.08 * sr)) / sr
    noise = ss.sosfilt(ss.butter(2, 7000, "hp", fs=sr, output="sos"), rng.normal(size=len(t)))
    return 0.3 * noise * np.exp(-t / decay)


def drum_events(seed: int, bars: int, bpm: float, loop: bool = False):
    """List of (time_s, kind). Deterministic per seed.

    Default: a song-like arrangement (intro, verse/chorus grooves, a break, fills and small
    per-bar variations), so only the true alignment matches everywhere.
    ``loop=True``: one identical bar repeated, i.e. genuinely ambiguous.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    verse_k = sorted(rng.choice(8, size=3, replace=False).tolist() + [0])
    chorus_k = sorted(set(rng.choice(8, size=4, replace=False).tolist() + [0]))
    events = []
    for bar in range(bars):
        t0 = bar * 4 * beat
        if loop:
            section = "verse"
        elif bar < 4:
            section = "intro"
        elif bars // 2 - 4 <= bar < bars // 2:
            section = "break"
        else:
            section = "chorus" if ((bar - 4) // 8) % 2 else "verse"
        kicks = {"verse": verse_k, "chorus": chorus_k, "intro": [], "break": [0]}[section]
        if not loop and section in ("verse", "chorus") and rng.random() < 0.35:
            kicks = sorted(set(kicks) | {int(rng.integers(1, 8))})  # small per-bar variation
        for e in range(8):  # eighth notes
            t = t0 + e * beat / 2
            if section != "break":
                events.append((t, "hat"))
            if e in kicks:
                events.append((t, "kick"))
            if e in (2, 6) and section != "break":
                events.append((t, "snare"))
        if not loop and bar % 4 == 3:  # fill: random sixteenth snares in the last beat
            for s16 in range(4):
                if rng.random() < 0.7:
                    events.append((t0 + 3 * beat + s16 * beat / 4, "snare"))
    return events


def render_drums(events, n, sr, seed, timbre=0):
    rng = np.random.default_rng(seed + 1000)
    kick = _kick(sr, decay=0.12 if timbre == 0 else 0.2, f0=55 if timbre == 0 else 62)
    out = np.zeros(n)
    for t, kind in events:
        i = int(round(t * sr))
        if kind == "kick":
            x = kick
        elif kind == "snare":
            x = _snare(
                sr, rng, decay=0.10 if timbre == 0 else 0.15, tone=190 if timbre == 0 else 220
            )
        else:
            x = _hat(sr, rng)
        j = min(n, i + len(x))
        if 0 <= i < n:
            out[i:j] += x[: j - i]
    return out


def make_song(seed=1, bars=64, bpm=110.0, sr=SR, loop=False):
    beat = 60.0 / bpm
    n = int((bars * 4 * beat + 2.0) * sr)
    events = drum_events(seed, bars, bpm, loop=loop)
    drums = render_drums(events, n, sr, seed)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed + 7)
    roots = np.full(bars, 49.0) if loop else rng.choice([41.2, 49.0, 55.0, 61.7], size=bars)
    bass = np.zeros(n)
    for bar, f in enumerate(roots):
        i0, i1 = int(bar * 4 * beat * sr), int((bar + 1) * 4 * beat * sr)
        seg = t[i0:i1] - t[i0]
        bass[i0:i1] = (
            0.3 * np.sin(2 * np.pi * f * seg) * (0.6 + 0.4 * np.cos(2 * np.pi * seg / beat))
        )
    pad = 0.08 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    return (drums + bass + pad).astype(np.float32), events


def make_recording(
    song_events,
    song_len,
    offset_s,
    duration_s,
    seed=1,
    sr=SR,
    gain=6.0,
    noise=0.05,
    tempo_scale=1.0,
):
    """Drum-only 'phone recording' whose t=0 is song time ``offset_s``.

    ``tempo_scale`` > 1 makes the player gradually rush (events land earlier over time).
    """
    n = int(duration_s * sr)
    shifted = [((t - offset_s) / tempo_scale, k) for t, k in song_events]
    drums = render_drums(shifted, n, sr, seed + 50, timbre=1)
    room = np.exp(-np.arange(int(0.15 * sr)) / (0.03 * sr)) * np.random.default_rng(
        seed + 9
    ).normal(size=int(0.15 * sr))
    room[0] = 1.0 / 0.05
    wet = ss.fftconvolve(drums, room * 0.05)[:n]
    rng = np.random.default_rng(seed + 3)
    return (gain * wet + noise * rng.normal(size=n)).astype(np.float32)
