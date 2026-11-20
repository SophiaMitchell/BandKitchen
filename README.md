# BandKitchen
A simple app for musicians to synchronize session video recordings with local audio files. 

Inspired by the fact that I'm a drummer who likes to drum along to Spotify. I also need practice with AI-assisted rapid tool prototyping, and this seemed like an interesting exercise.

## What it does (so far)
- **Auto-sync:** finds where your play-along video sits in the original song, with no manual
  waveform dragging. It flags low confidence instead of guessing.
- **Drift report:** shows where you rushed or dragged against the track, as a practice
  diagnostic. It is never applied to the mix.
- **Loudness-based mixing:** normalizes the song and your recording by loudness (LUFS), puts
  your instrument a set number of dB under the song, and limits peaks. A blown-out phone mic
  no longer clips the export.
- **Render:** copies your video untouched and replaces the audio with the mix.

Planned: play-along splicing (remove your instrument from the song with Demucs), an mp3
library, presets, and a retro Java desktop UI with live mixing.

## Quick start
Requires Python 3.10+ and ffmpeg (`brew install ffmpeg`).

```bash
pip install -e ".[dev]"
bandkitchen sync TheManFullSingle.mov "02 - The Man.mp3" --drift-report
bandkitchen render TheManFullSingle.mov "02 - The Man.mp3" -o TheMan.mp4 --instrument-db -3
```

`--instrument-db` sets your instrument's level relative to the song (negative means quieter).
`--offset` overrides auto-sync. Add `--json` to any command for machine-readable output.

## Development
```bash
pytest                       # synthetic and end-to-end tests
BANDKITCHEN_MEDIA_DIR=~/BandKitchenMedia pytest -m local_media   # your own recordings
```
No songs or recordings are committed (the repo is public). See `tests/local_cases.json`.
