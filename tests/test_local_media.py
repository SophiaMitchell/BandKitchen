"""Regression tests on real recordings. Skipped unless BANDKITCHEN_MEDIA_DIR is set.

Media never goes in the repo (copyright, size). See tests/local_cases.json.
"""

import json
import os
from pathlib import Path

import pytest

from bandkitchen import engine

MEDIA_DIR = os.environ.get("BANDKITCHEN_MEDIA_DIR")
CASES = json.loads((Path(__file__).parent / "local_cases.json").read_text())["cases"]

pytestmark = [
    pytest.mark.local_media,
    pytest.mark.skipif(not MEDIA_DIR, reason="set BANDKITCHEN_MEDIA_DIR to run"),
]


def _find(pattern: str) -> Path:
    hits = sorted(Path(MEDIA_DIR).glob(pattern))
    if not hits:
        pytest.skip(f"no file matching {pattern!r} in {MEDIA_DIR}")
    return hits[0]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_auto_sync_matches_hand_sync(case):
    rep = engine.sync_files(_find(case["video"]), _find(case["song"]))
    assert rep.reliable
    assert rep.result.offset_s == pytest.approx(case["expected_offset_s"], abs=case["tolerance_s"])
