"""Command-line interface. Every command can emit JSON (--json) so the UI can drive it.

bandkitchen sync VIDEO SONG [--drift-report] [--json]
bandkitchen render VIDEO SONG -o OUT.mp4 [--instrument-db -3] [--song-lufs -14]
                   [--offset SECONDS] [--json]
bandkitchen loudness FILE [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, engine, media, mix, stems
from .loudness import integrated_lufs


def _emit(data: dict, as_json: bool, text: str) -> None:
    print(json.dumps(data, indent=2) if as_json else text)


def _cmd_sync(a: argparse.Namespace) -> int:
    rep = engine.sync_files(a.video, a.song, drift_report=a.drift_report)
    r = rep.result
    lines = [
        f"offset: {r.offset_s:+.3f} s  (song time at video start)",
        f"correlation {r.correlation:.2f}, peak margin {r.peak_margin:.2f}"
        + ("" if rep.reliable else "  ** LOW CONFIDENCE: check the song matches **"),
    ]
    for p in rep.drift:
        bar = "#" * min(int(abs(p.deviation_ms) / 5), 30)
        side = "early" if p.deviation_ms > 0 else "late "
        lines.append(f"  {p.video_time_s:6.1f}s  {p.deviation_ms:+6.1f} ms {side} {bar}")
    _emit(rep.to_dict(), a.json, "\n".join(lines))
    return 0 if rep.reliable else 2


def _cmd_render(a: argparse.Namespace) -> int:
    settings = mix.MixSettings(
        song_lufs=a.song_lufs, instrument_rel_db=a.instrument_db, ceiling_dbfs=a.ceiling
    )
    sep = stems.make_separator(a.separator, a.remove) if a.remove else None
    rep = engine.render_files(
        a.video,
        a.song,
        a.output,
        settings,
        offset_s=a.offset,
        remove=a.remove,
        separator=sep,
        use_cache=not a.no_cache,
    )
    m = rep.mix
    removed = ""
    if rep.removed:
        how = "cached" if rep.stems and rep.stems.cached else "separated"
        removed = f"removed {rep.removed} from the song ({rep.stems.separator}, {how})\n"
    text = removed + (
        f"wrote {rep.output}\n"
        f"offset {rep.offset_s:+.3f} s" + (" (auto)" if rep.sync else " (manual)") + "\n"
        f"song {m.song_lufs_in:.1f} LUFS -> gain {m.song_gain_db:+.1f} dB; "
        f"instrument {m.instrument_lufs_in:.1f} LUFS -> gain {m.instrument_gain_db:+.1f} dB\n"
        f"limiter max reduction {m.limiter_max_reduction_db:.1f} dB; "
        f"output {m.output_lufs:.1f} LUFS, peak {m.output_peak_dbfs:.1f} dBFS"
    )
    if rep.sync and not rep.sync.reliable:
        text += "\n** LOW SYNC CONFIDENCE: check the song matches, or pass --offset **"
    _emit(rep.to_dict(), a.json, text)
    return 0


def _cmd_stems(a: argparse.Namespace) -> int:
    sep = stems.make_separator(a.separator, a.instrument)
    _, rep = engine.separate_song(a.song, sep, use_cache=True)
    _emit(
        rep.to_dict(),
        a.json,
        f"{'already cached' if rep.cached else 'separated and cached'}: "
        f"{', '.join(rep.stems)} ({rep.separator})",
    )
    return 0


def _cmd_loudness(a: argparse.Namespace) -> int:
    x = media.load_audio(a.file, mix.RENDER_SR)
    lufs = integrated_lufs(x, mix.RENDER_SR)
    peak = float(abs(x).max())
    _emit(
        {"file": a.file, "lufs": lufs, "sample_peak": peak},
        a.json,
        f"{lufs:.1f} LUFS, sample peak {peak:.2f} ({'over' if peak > 1 else 'within'} full scale)",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bandkitchen", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="find the song offset for a play-along video")
    s.add_argument("video")
    s.add_argument("song")
    s.add_argument(
        "--drift-report",
        action="store_true",
        help="show where you rush/drag vs the song (diagnostic; never applied)",
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_sync)

    r = sub.add_parser("render", help="sync, mix and write the combined video")
    r.add_argument("video")
    r.add_argument("song")
    r.add_argument("-o", "--output", required=True)
    r.add_argument(
        "--instrument-db",
        type=float,
        default=mix.MixSettings.instrument_rel_db,
        help="your audio's loudness relative to the song, dB (default %(default)s)",
    )
    r.add_argument(
        "--song-lufs",
        type=float,
        default=mix.MixSettings.song_lufs,
        help="song loudness target (default %(default)s)",
    )
    r.add_argument(
        "--ceiling",
        type=float,
        default=mix.MixSettings.ceiling_dbfs,
        help="limiter ceiling, dBFS (default %(default)s)",
    )
    r.add_argument(
        "--offset",
        type=float,
        default=None,
        help="skip auto-sync and use this offset (song time at video start), s",
    )
    r.add_argument(
        "--remove",
        choices=stems.INSTRUMENTS,
        default=None,
        help="take this instrument out of the song first (play-along splicing)",
    )
    r.add_argument(
        "--separator",
        choices=("demucs", "hpss"),
        default="demucs",
        help="demucs: best quality (needs the stems extra); "
        "hpss: rough drums-only fallback, no extra installs",
    )
    r.add_argument("--no-cache", action="store_true", help="re-separate even if cached")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_render)

    st = sub.add_parser("stems", help="separate a song into stems ahead of time (cached)")
    st.add_argument("song")
    st.add_argument(
        "--instrument",
        choices=stems.INSTRUMENTS,
        default="drums",
        help="picks the model: guitar/piano need the 6-stem model",
    )
    st.add_argument("--separator", choices=("demucs", "hpss"), default="demucs")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=_cmd_stems)

    lo = sub.add_parser("loudness", help="measure integrated loudness and peak of a file")
    lo.add_argument("file")
    lo.add_argument("--json", action="store_true")
    lo.set_defaults(func=_cmd_loudness)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (media.MediaError, stems.SeparatorUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
