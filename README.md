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

- **Play-along splicing:** removes your instrument from the song with Demucs, so your
  part replaces the original. Separated songs are cached, so later renders are quick.

Planned: an mp3 library, presets, and a retro Java desktop UI with live mixing.

## Quick start
Requires Python 3.10–3.14 and ffmpeg (`brew install ffmpeg`). Stem separation depends on
PyTorch, which doesn't support Python 3.15 yet.

### Set up a virtual environment
**If your default Python is older than 3.10 or newer than 3.14, you must do this step.** It
pins the project to a supported version. macOS ships Python 3.9, so this applies to most Macs.

```bash
brew install python@3.12
cd BandKitchen
python3.12 -m venv .venv
source .venv/bin/activate          # run in each new terminal; prompt shows (.venv)
python --version                   # should say 3.12.x
python -m pip install -e ".[stems,dev]"
```

The install creates the `bandkitchen` command inside `.venv`, so it works whenever the venv is
active. Do it once per new venv, or again when `pyproject.toml` changes. If `python --version`
doesn't say 3.12, delete `.venv` and recreate it with the full path
`/opt/homebrew/bin/python3.12 -m venv .venv`.

`stems` installs Demucs (for removing your instrument from the song) and `dev` installs the
test and lint tools. Both are optional.

### Run it
```bash
bandkitchen sync TheManFullSingle.mov "02 - The Man.mp3" --drift-report
bandkitchen render TheManFullSingle.mov "02 - The Man.mp3" -o TheMan.mp4 --instrument-db -3
```

To replace the band's drummer with you, add `--remove drums`:

```bash
bandkitchen render HarderToBreatheSingle.mov "01 - Harder To Breathe.mp3" -o HTB.mp4 --remove drums
```

The first run of a song downloads the Demucs model (about 80 MB) and separates the song, which
takes around a minute. After that, the stems are cached in `~/.bandkitchen-cache`. `--remove`
also takes `bass` and `vocals`, plus `guitar` and `piano` (these two use a 6-stem model and
separate less cleanly). `--separator hpss` is a rough, drums-only fallback that needs no
PyTorch.

`--instrument-db` sets your instrument's level relative to the song (negative means quieter).
The default is −3 dB, or −2 dB with `--remove`. `--offset` overrides auto-sync. Add `--json` to
any command for machine-readable output.

## Development
```bash
pytest                       # synthetic and end-to-end tests
BANDKITCHEN_MEDIA_DIR=~/BandKitchenMedia pytest -m local_media   # your own recordings
```
No songs or recordings are committed (the repo is public). See `tests/local_cases.json`.
