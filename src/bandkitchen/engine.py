"""File-level operations shared by the CLI and (later) the HTTP API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import media, mix, render, sync

# Below these, the match is suspect: likely a different song or recording, or too little overlap.
MIN_CORRELATION = 0.25
MIN_PEAK_MARGIN = 0.20


@dataclass
class SyncReport:
    video: str
    song: str
    result: sync.SyncResult
    reliable: bool
    drift: list[sync.DriftPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "song": self.song,
            **self.result.to_dict(),
            "reliable": self.reliable,
            "drift": [p.to_dict() for p in self.drift],
        }


def sync_files(video: str | Path, song: str | Path, drift_report: bool = False) -> SyncReport:
    v = media.load_audio(video, sync.ANALYSIS_SR, mono=True)
    s = media.load_audio(song, sync.ANALYSIS_SR, mono=True)
    result = sync.estimate_offset(v, s)
    reliable = result.correlation >= MIN_CORRELATION and result.peak_margin >= MIN_PEAK_MARGIN
    drift = sync.measure_drift(v, s, result.offset_s) if drift_report else []
    return SyncReport(str(video), str(song), result, reliable, drift)


@dataclass
class RenderReport:
    output: str
    offset_s: float
    settings: mix.MixSettings
    mix: mix.MixReport
    sync: SyncReport | None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "offset_s": self.offset_s,
            "settings": self.settings.to_dict(),
            "mix": self.mix.to_dict(),
            "sync": self.sync.to_dict() if self.sync else None,
        }


def render_files(
    video: str | Path,
    song: str | Path,
    out: str | Path,
    settings: mix.MixSettings | None = None,
    offset_s: float | None = None,
) -> RenderReport:
    """Sync (unless ``offset_s`` is given), mix, and write ``out``."""
    sync_report = None
    if offset_s is None:
        sync_report = sync_files(video, song)
        offset_s = sync_report.result.offset_s
    sr = mix.RENDER_SR
    inst = mix.to_stereo(media.load_audio(video, sr))
    song_aligned = mix.align_song(media.load_audio(song, sr), offset_s, len(inst), sr)
    settings = settings or mix.MixSettings()
    mixed, mix_report = mix.mix(inst, song_aligned, sr, settings)
    render.mux(video, mixed, sr, out)
    return RenderReport(str(out), offset_s, settings, mix_report, sync_report)
