import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from degas_worker.video import Audio, encode_mp4

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def streams(data: bytes, tmp_path: Path) -> dict[str, float]:
    """Each stream's duration, and the video's frame count as `frames`."""
    path = tmp_path / "out.mp4"
    path.write_bytes(data)
    probe = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,duration,nb_frames"]
    out = subprocess.run(
        [*probe, "-of", "json", str(path)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    found = json.loads(out)["streams"]
    durations = {s["codec_name"]: float(s["duration"]) for s in found}
    return {
        **durations,
        "frames": float(next(s for s in found if s["codec_name"] == "h264")["nb_frames"]),
    }


def test_encode_without_sound(tmp_path: Path) -> None:
    frames = [bytes(16 * 16 * 3)] * 24
    assert set(streams(encode_mp4(frames, 16, 16, 24), tmp_path)) == {"h264", "frames"}


def test_encode_with_stereo_sound(tmp_path: Path) -> None:
    rate = 24000
    pcm = struct.pack(f"<{2 * rate}f", *([0.1, -0.1] * rate))  # one second, two channels
    frames = [bytes(16 * 16 * 3)] * 24
    found = streams(encode_mp4(frames, 16, 16, 24, audio=Audio(pcm, rate, 2)), tmp_path)
    assert set(found) == {"h264", "aac", "frames"}
    assert found["aac"] == pytest.approx(1.0, abs=0.05)


def test_short_sound_keeps_every_frame(tmp_path: Path) -> None:
    # LTX's audio comes out a little shorter than its 8k + 1 frames: the last frame (the one a
    # last-frame condition sets) must stay.
    rate = 24000
    pcm = struct.pack(f"<{rate * 5}f", *([0.0] * (rate * 5)))  # 5.00 s against 5.04 s of video
    frames = [bytes(16 * 16 * 3)] * 121
    found = streams(encode_mp4(frames, 16, 16, 24, audio=Audio(pcm, rate)), tmp_path)
    assert found["frames"] == 121
