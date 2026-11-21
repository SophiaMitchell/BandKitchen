# BandKitchen
A simple app for musicians to synchronize session video recordings with local audio files. 

Inspired by the fact that I'm a drummer who likes to drum along to Spotify. I also need practice with AI-assisted rapid tool prototyping, and this seemed like an interesting exercise.

### What it does (so far)
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

# Quick start
Requires Python 3.10–3.14 and ffmpeg (`brew install ffmpeg`). Stem separation depends on
PyTorch, which doesn't support Python 3.15 yet.

## Set Up a Virtual Environment
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

`stems` installs Demucs (model for removing your instrument from the song) and `dev` installs the
test and lint tools. Both are optional.

## Command Reference

### `bandkitchen [FUNCTION] [VIDEO] [SONG] -o [OPTIONS]`

| Input | Description |
|:------------------|:---|
| `FUNCTION` | Select a main function for bandkitchen (see list below) |
| `VIDEO` | Your play-along video recording (.mov, .mp4, .m4v...) |
| `SONG` | The original song (.mp3, .m4a, .wav, .flac, .aiff...) |
| `-o`, `--output` | The video file to write. `.mp4` recommended; `.mov` also works |
| `OPTIONS` | See table below |

Functions:
  * `sync` : Syncs the video and song without writing any files. Prints offset in seconds (song time at the video's first frame; negative means the video starts before the song) and confidence values (how good the synchronization is). Useful for practicing.\
            `bandkitchen sync mydrums.mov hitsong.mp3 --drift-report`
    
* `render` : Syncs, mixes and writes the combined video. Requires an output file name. Outputs a video file, with your original video stream copied untouched (no re-encoding) and the audio replaced by the mix (AAC, 256 kbps, 48 kHz stereo). Audio is trimmed and/or padded to result in a video the same length as your original video. The command also prints the offset used, the gain applied to each track and the final loudness and peak. Low sync confidence prints a warning but still renders.\
            `bandkitchen render mydrums.mov hitsong.mp3 -o mydrumalong.mp4`
  
* `stems ` : Separates a song ahead of time, so a later `render --remove` skips that step. Only requires a song input. Stems are written to the local cache. Alerts if they were already cached.\
            `bandkitchen stems hitsong.mp3`
  
* `loudness` : Measures a file's loudness in dB. More negative is more quiet. Prints integrated loudness (LUFS) and sample peak, noting if the file goes over full scale.\
            `bandkitchen loudness mydrums.mov --json` 

| Option | Default | Description |
|:---------------------------|:-----:|:---|
| `--drift-report` | off | Shows where you rushed or dragged against the song (see [Reading the drift report](#reading-the-drift-report)). Never applied to the render |
| `--remove INSTRUMENT` | none | Take an instrument out of the song file (see [Removing Instruments](#removing-instruments)) |
| `--separator` | `demucs` | Select the instrument separator tool to use (goes with `--remove`), `demucs` (best quality, needs the `stems` extra) or `hpss` (rough, drums only) |
| `--instrument INSTRUMENT` | `drums` | Picks the demucs model to use if using it for separation: `guitar` or `piano` use the 6-stem one; anything else uses the 4-stem one |
| `--no-cache` | off | Separate the song again even if its stems are cached |
| `--instrument-db DB` | −3, or −2 with `--remove` | Your recording's loudness relative to the song, in dB. Negative is quieter |
| `--song-lufs LUFS` | −14 | Loudness target for the song |
| `--ceiling DBFS` | −1 | Limiter ceiling: the loudest any sample can be |
| `--offset SECONDS` | auto | Skip auto-sync and use this offset |
| `--json` | off | Print the result as JSON |

Exit codes: `0` success, `1` error (missing file, bad option, Demucs not installed), `2` sync confidence too low (`sync` only).

### Environment variables
| Variable | Default | Description |
|:---------------------:|:---:|:---|
| `BANDKITCHEN_CACHE` | `~/.bandkitchen-cache` | Where separated stems are stored. Safe to delete; stems are recreated when needed |
| `BANDKITCHEN_MEDIA_DIR` | unset | Folder of your own recordings, for `pytest -m local_media` |

### For developers
```bash
pytest                       # synthetic and end-to-end tests
BANDKITCHEN_MEDIA_DIR=~/BandKitchenMedia pytest -m local_media   # your own recordings
```
No songs or recordings are committed since the repo is public and I don't like fighting with lawyers. See `tests/local_cases.json`.

# Removing Instruments
The first time `--remove` is run on an .mp3, this tool downloads the Demucs model (about 80 MB) and separates the
song, which takes around a minute. After that, the stems are cached locally so subsequent renders are quick (~15/sec).

| `--remove` | Takes out of the song | Separation quality |
|---|---|---|
| `drums` | The full kit: kick, snare, toms, cymbals | Very clean |
| `bass` | Bass guitar, synth bass | Very clean |
| `vocals` | Lead and backing vocals | Clean |
| `guitar` | Electric and acoustic guitars | Usable, some bleed (6-stem model) |
| `piano` | Piano and keys | Usable, some bleed (6-stem model) |
| `other` | Everything that isn't drums, bass or vocals (guitars, keys, synths, strings) | Clean |

`guitar` and `piano` switch automatically to Demucs' 6-stem model, which separates those two
less cleanly than the rest. 

## Alternative to Demucs
If you don't want to use Demucs, the option `--separator hpss` is a fallback. Only `drums` and `other`
work, and `drums` removes snare and cymbals far better than the kick. So not great, but can get the job 
done if you're a drummer.

# Reading the Drift Report
Auto-sync lines your whole video up with the song using one fixed offset. It never stretches
or nudges your playing, so your real timing stays audible. `--drift-report` then shows how far
your playing sat from that line through the song, as a practice tool:

```
offset: -3.442 s  (song time at video start)
correlation 0.37, peak margin 0.38
    16.0s    +5.1 ms early #
    24.0s    -8.2 ms late  #
    40.0s   +15.5 ms early ###
    56.0s   -14.1 ms late  ##
```

- **Time (left):** the middle of each stretch analysed, in *video* time, so you can scrub to it.
  Each line looks at 16 s of playing, and lines are 8 s apart, so neighbouring lines overlap.
- **ms early / late:** how far ahead of the record (early, positive: rushing) or behind it
  (late, negative: dragging) you were on average in that stretch, compared with your average
  across the whole take.
- **The bar:** one `#` per 5 ms, for spotting the biggest moments at a glance.

How to interpret it:
- **Within about ±10 ms is tight.** Two drummers never hit at exactly the same moment, and the
  record's drummer has their own feel too, so small numbers are normal.
- **Look for runs, not single lines.** Several lines drifting the same way (for example +20,
  +25, +30) means you're steadily speeding up through that section. A one-off spike often
  comes from a fill or a section where the parts differ.
- **Sparse sections read less reliably.** Breaks, intros or anywhere with little to match can
  produce odd numbers. With `--json`, each point includes a `strength` value; low strength
  means there was little to compare.
- Each point is only searched within ±150 ms of the overall offset, so a report can't jump a
  whole beat.

The report never changes the render. Drift correction, if it's ever added, will be a separate
option that's off by default.
