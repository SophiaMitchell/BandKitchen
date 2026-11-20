"""Write the final video: the original video stream copied untouched, plus the new mix.

The video is never re-encoded or re-timed (iPhone footage is often variable frame rate;
re-timing it causes drift and quality loss). Only the audio track is replaced.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .media import MediaError, _require, probe


def mux(
    video_path: str | Path,
    audio: np.ndarray,
    sr: int,
    out_path: str | Path,
    audio_bitrate: str = "256k",
) -> Path:
    """Copy ``video_path``'s video stream and replace its audio with ``audio`` (float, (n, 2))."""
    info = probe(video_path)
    if not info.has_video:
        raise MediaError(f"{video_path} has no video stream")
    out_path = Path(out_path)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    cmd = [
        _require("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(audio.shape[1]),
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        "-movflags",
        "+faststart",
    ]
    if info.video_codec == "hevc" and out_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        cmd += ["-tag:v", "hvc1"]  # so QuickTime/iOS will play HEVC in an MP4
    cmd.append(str(out_path))
    proc = subprocess.run(cmd, input=audio.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg mux failed: {proc.stderr.decode().strip()}")
    return out_path
