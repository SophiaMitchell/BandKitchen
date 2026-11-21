"""End-to-end: synthesize a song file and a 'phone video' with ffmpeg, then run the CLI."""

import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from bandkitchen import cli, media
from bandkitchen.loudness import integrated_lufs
from tests.synth import SR, make_recording, make_song

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

OFFSET = 7.25


def _write_wav(path, x, sr):
    x16 = np.clip(x / max(np.abs(x).max(), 1e-9), -1, 1)  # WAV is just a carrier here
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x16 * 32767).astype("<i2").tobytes())


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    audio, events = make_song(seed=5, bars=48)
    rec = make_recording(events, len(audio), OFFSET, duration_s=40.0)
    song_wav, rec_wav = d / "song.wav", d / "rec.wav"
    _write_wav(song_wav, audio, SR)
    _write_wav(rec_wav, rec, SR)
    song_mp3 = d / "song.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", song_wav, "-b:a", "192k", song_mp3], check=True
    )
    video = d / "take.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30",
            "-i",
            rec_wav,
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            video,
        ],
        check=True,
    )
    return video, song_mp3, d


def test_cli_sync_json(files, capsys):
    video, song, _ = files
    code = cli.main(["sync", str(video), str(song), "--json", "--drift-report"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["reliable"] is True
    assert out["offset_s"] == pytest.approx(OFFSET, abs=0.015)
    assert len(out["drift"]) > 0


def test_cli_render_copies_video_and_levels_audio(files, capsys):
    video, song, d = files
    out_path = d / "out.mp4"
    code = cli.main(["render", str(video), str(song), "-o", str(out_path), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 0 and out_path.exists()
    info_in, info_out = media.probe(video), media.probe(out_path)
    assert info_out.has_video and info_out.video_codec == info_in.video_codec
    assert info_out.duration_s == pytest.approx(info_in.duration_s, abs=0.1)
    x = media.load_audio(out_path, 48000)
    assert integrated_lufs(x, 48000) == pytest.approx(rep["mix"]["output_lufs"], abs=0.5)
    assert np.abs(x).max() < 1.0  # no clipping after AAC encode


def test_cli_render_manual_offset(files, capsys):
    video, song, d = files
    code = cli.main(
        ["render", str(video), str(song), "-o", str(d / "m.mp4"), "--offset", "7.0", "--json"]
    )
    rep = json.loads(capsys.readouterr().out)
    assert code == 0 and rep["offset_s"] == 7.0 and rep["sync"] is None


def test_cli_missing_file_is_clean_error(tmp_path, capsys):
    code = cli.main(["loudness", str(tmp_path / "nope.mp3")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_render_remove_drums_hpss(files, capsys, tmp_path, monkeypatch):
    """End-to-end splicing with the no-install fallback separator."""
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path / "cache"))
    video, song, d = files
    out_path = d / "nodrums.mp4"
    args = [
        "render",
        str(video),
        str(song),
        "-o",
        str(out_path),
        "--json",
        "--remove",
        "drums",
        "--separator",
        "hpss",
    ]
    assert cli.main(args) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["removed"] == "drums"
    assert rep["stems"]["cached"] is False
    assert rep["settings"]["instrument_rel_db"] == -2.0
    assert rep["offset_s"] == pytest.approx(OFFSET, abs=0.015)
    assert media.probe(out_path).has_video
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["stems"]["cached"] is True


def test_cli_render_explicit_level_overrides_spliced_default(files, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path / "cache"))
    video, song, d = files
    cli.main(
        [
            "render",
            str(video),
            str(song),
            "-o",
            str(d / "x.mp4"),
            "--json",
            "--remove",
            "drums",
            "--separator",
            "hpss",
            "--instrument-db",
            "-5",
        ]
    )
    assert json.loads(capsys.readouterr().out)["settings"]["instrument_rel_db"] == -5.0


def test_cli_hpss_rejects_non_drums_before_any_work(files, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path / "cache"))
    video, song, d = files
    for args in (
        [
            "render",
            str(video),
            str(song),
            "-o",
            str(d / "x.mp4"),
            "--remove",
            "bass",
            "--separator",
            "hpss",
        ],
        ["stems", str(song), "--remove", "other", "--separator", "hpss"],
    ):
        assert cli.main(args) == 1
        assert "only remove drums" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists()  # nothing was separated


def test_cli_stems_then_render_uses_the_cache(files, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("BANDKITCHEN_CACHE", str(tmp_path / "cache"))
    video, song, d = files
    assert cli.main(["stems", str(song), "--remove", "drums", "--separator", "hpss", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cached"] is False
    assert (
        cli.main(
            [
                "render",
                str(video),
                str(song),
                "-o",
                str(d / "y.mp4"),
                "--json",
                "--remove",
                "drums",
                "--separator",
                "hpss",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["stems"]["cached"] is True
