# BandKitchen — instructions for coding agents

BandKitchen syncs a video of a musician playing along to a song with the original .mp3, balances
the mix, optionally removes the player's instrument from the song ("play-along splicing"), and
renders a shareable video. Owner: a drummer and senior engineer; treat her as the reviewer.

## Architecture (route A: Mac first)
- **Python engine** (`src/bandkitchen/`) — all audio/DSP. CLI first; every command has `--json`.
  A thin local HTTP API (FastAPI, planned) will wrap the same `engine.py` functions.
- **Java UI** (JavaFX, planned, `ui/`) — a thin client. Live preview plays the video plus aligned
  WAV stems and mixes them in real time; it only asks the engine to render on Save.
- iOS is deferred. Keep the engine behind a clean CLI/API contract so a future Swift client could
  talk to it.

Pipeline modules: `media` (ffmpeg decode → float32, probe) → `sync` (offset + drift report)
→ `mix` (LUFS gain staging, balance, limiter) → `render` (mux). Planned: `stems` (Demucs),
`library` (mp3 index + analysis cache keyed by file hash), presets.

## Firm rules (do not change without the owner's say-so)
1. **Float32 until the final limiter.** Phone audio is often stored far above full scale
   (e.g. +5 LUFS, +18 dBFS peaks). Never clip or convert to int before `mix.limit`.
2. **Set levels by loudness (LUFS), not percentages.** Presets are "instrument dB relative to
   the song".
3. **Never re-encode or re-time the video.** Copy the video stream; replace only audio.
   iPhone footage can be variable frame rate.
4. **Drift correction is opt-in and defaults to off** (not implemented yet). One global offset
   keeps the player's real timing audible. `sync.measure_drift` is a diagnostic only.
5. **Offset convention:** `song_time = video_time + offset_s`. Positive = video starts partway
   into the song.
6. **No copyrighted media in the repo.** It is public. Tests use synthetic audio
   (`tests/synth.py`); real recordings are local-only (`tests/local_cases.json`).
7. Report low sync confidence instead of guessing (`engine.MIN_PEAK_MARGIN`,
   `MIN_CORRELATION`).

## Commands
```bash
pip install -e ".[dev]"            # needs ffmpeg on PATH (brew install ffmpeg)
pytest                              # synthetic + end-to-end tests (~30 s)
BANDKITCHEN_MEDIA_DIR=~/BandKitchenMedia pytest -m local_media   # real recordings
ruff check . && ruff format --check .
bandkitchen sync VIDEO SONG --drift-report
bandkitchen render VIDEO SONG -o out.mp4 --instrument-db -3
```

## Conventions
- Python ≥ 3.10, numpy/scipy core. librosa/demucs are optional extras for features that need
  them; keep the core importable without them.
- Pure functions on numpy arrays in DSP modules; file I/O only in `media`, `render`, `engine`.
- Every DSP change needs a test with a known answer (synthetic signal with a known offset,
  level, etc.). If you change sync, also run the local-media test if media is available.
- Small PRs, one concern each. Explain DSP choices in docstrings (why, not just what).

## Known limits / open questions
- Real-world sync precision is about ±10–20 ms (two drummers never hit at the same time); the
  UI will offer a nudge. Auto-sync currently lands ~13 ms off the owner's by-ear sync on
  The Man.
- Short excerpts from the middle of a song can match the wrong section; they are flagged by a
  low `peak_margin`.
