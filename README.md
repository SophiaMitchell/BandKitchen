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
bandkitchen render TheManFullSingle.mov "02 - The Man.mp3" -o TheMan.mp4
bandkitchen render HarderToBreatheSingle.mov "01 - Harder To Breathe.mp3" -o HTB.mp4 --remove drums
```

The first `--remove` run on a song downloads the Demucs model (about 80 MB) and separates the
song, which takes around a minute. After that, the stems are cached and later renders are quick.
Tip: drag a file into Terminal to paste its full path, quoted correctly.

## Instruments you can remove
Pass one of these to `--remove`:

| `--remove` | Takes out of the song | Separation quality |
|---|---|---|
| `drums` | The full kit: kick, snare, toms, cymbals | Very clean |
| `bass` | Bass guitar, synth bass | Very clean |
| `vocals` | Lead and backing vocals | Clean |
| `guitar` | Electric and acoustic guitars | Usable, some bleed (6-stem model) |
| `piano` | Piano and keys | Usable, some bleed (6-stem model) |
| `other` | Everything that isn't drums, bass or vocals (guitars, keys, synths, strings) | Clean |

`guitar` and `piano` switch automatically to Demucs's 6-stem model, which separates those two
less cleanly than the rest. With `--separator hpss` (the no-PyTorch fallback), only `drums` and `other`
work, and `drums` removes snare and cymbals far better than the kick.

## Command reference
Every command takes `--json` to print a machine-readable result instead of text, and `-h` to
show its help. Exit codes: `0` success, `1` error (missing file, bad option, Demucs not
installed), `2` sync confidence too low (`sync` only).

### `bandkitchen sync VIDEO SONG`
Finds where the video sits in the song, without writing any files.

| Input | Description |
|---|---|
| `VIDEO` | Your play-along recording (.mov, .mp4, .m4v...). Needs a video and an audio track |
| `SONG` | The original song (.mp3, .m4a, .wav, .flac, .aiff...) |

| Option | Default | Description |
|---|---|---|
| `--drift-report` | off | Show where you rushed or dragged, every 8 s. A diagnostic only; never applied |
| `--json` | off | Print the result as JSON |

**Output:** the offset in seconds (song time at the video's first frame; negative means the
video starts before the song), the confidence numbers, and optionally the drift report.

### `bandkitchen render VIDEO SONG -o OUTPUT`
Syncs, mixes and writes the combined video.

| Input | Description |
|---|---|
| `VIDEO` | Your play-along recording (as for `sync`) |
| `SONG` | The original song (as for `sync`) |
| `-o`, `--output` | The video file to write. `.mp4` recommended; `.mov` also works |

| Option | Default | Description |
|---|---|---|
| `--remove INSTRUMENT` | none | Take an instrument out of the song first (see the table above) |
| `--instrument-db DB` | −3, or −2 with `--remove` | Your recording's loudness relative to the song, in dB. Negative is quieter |
| `--song-lufs LUFS` | −14 | Loudness target for the song |
| `--ceiling DBFS` | −1 | Limiter ceiling: the loudest any sample can be |
| `--offset SECONDS` | auto | Skip auto-sync and use this offset |
| `--separator` | `demucs` | `demucs` (best, needs the `stems` extra) or `hpss` (rough, drums only) |
| `--no-cache` | off | Separate the song again even if its stems are cached |
| `--json` | off | Print the result as JSON |

**Output:** the video file, with your original video stream copied untouched (no re-encoding)
and the audio replaced by the mix (AAC, 256 kbps, 48 kHz stereo). It is the same length as
your video. The command also prints the offset used, the gain applied to each track and the
final loudness and peak. Low sync confidence prints a warning but still renders.

### `bandkitchen stems SONG`
Separates a song ahead of time, so a later `render --remove` skips that step.

| Input | Description |
|---|---|
| `SONG` | The original song |

| Option | Default | Description |
|---|---|---|
| `--instrument INSTRUMENT` | `drums` | Picks the model: `guitar` or `piano` use the 6-stem one; anything else uses the 4-stem one |
| `--separator` | `demucs` | `demucs` or `hpss` |
| `--json` | off | Print the result as JSON |

**Output:** stems written to the cache; prints whether they were already cached.

### `bandkitchen loudness FILE`
Measures a file's loudness.

| Input | Description |
|---|---|
| `FILE` | Any audio or video file |

**Output:** integrated loudness (LUFS) and sample peak, noting if the file goes over full scale.
Option: `--json`.

### Environment variables
| Variable | Default | Description |
|---|---|---|
| `BANDKITCHEN_CACHE` | `~/.bandkitchen-cache` | Where separated stems are stored. Safe to delete; stems are recreated when needed |
| `BANDKITCHEN_MEDIA_DIR` | unset | Folder of your own recordings, for `pytest -m local_media` |

## Development
```bash
pytest                       # synthetic and end-to-end tests
BANDKITCHEN_MEDIA_DIR=~/BandKitchenMedia pytest -m local_media   # your own recordings
```
No songs or recordings are committed (the repo is public). See `tests/local_cases.json`.
