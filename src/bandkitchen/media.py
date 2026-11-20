"""Media I/O via ffmpeg. All audio enters the engine as float32 and is never clipped here.

Phone recordings are often stored far above full scale (AAC decodes to floats > 1.0).
Decoding to f32le preserves that headroom so the mix stage can turn it down losslessly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise MediaError(f"{tool} not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`).")
    return path


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    has_video: bool
    video_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None


def probe(path: str | Path) -> MediaInfo:
    out = subprocess.run(
        [
            _require("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,sample_rate,channels,disposition",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    # Cover art in an mp3 shows up as an attached_pic video stream; that is not real video.
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    return MediaInfo(
        duration_s=float(info["format"]["duration"]),
        has_video=video is not None,
        video_codec=video.get("codec_name") if video else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio else None,
        audio_channels=int(audio["channels"]) if audio else None,
    )


def load_audio(path: str | Path, sr: int, mono: bool = False) -> np.ndarray:
    """Decode a file's first audio stream to float32.

    Returns shape (n,) if mono else (n, 2). Values may exceed [-1, 1]; that is intentional.
    """
    channels = 1 if mono else 2
    out = subprocess.run(
        [
            _require("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            str(channels),
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
    )
    if out.returncode != 0:
        raise MediaError(f"ffmpeg failed to decode {path}: {out.stderr.decode().strip()}")
    x = np.frombuffer(out.stdout, dtype=np.float32)
    return x if mono else x.reshape(-1, 2)
