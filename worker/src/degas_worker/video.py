"""MP4 encoding with ffmpeg (preinstalled on Colab). Free of torch so it can be tested anywhere.

H.264 in yuv420p with the index at the front, so it plays inline on iOS (design §4.3), and
AAC for a model that makes sound (LTX-2).
"""

import subprocess
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple


class EncodeError(RuntimeError):
    pass


class Audio(NamedTuple):
    pcm: bytes  # float32 little-endian samples, interleaved when there are several channels
    rate: int
    channels: int = 1


def encode_mp4(
    frames: Iterable[bytes],
    width: int,
    height: int,
    fps: float,
    crf: int = 18,
    audio: Audio | None = None,
) -> bytes:
    """Encode raw RGB24 frames (`width * height * 3` bytes each), with `audio` if given."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        sound: list[str] = []
        if audio is not None:
            pcm = Path(tmp) / "audio.f32"
            pcm.write_bytes(audio.pcm)
            sound = ["-f", "f32le", "-ar", str(audio.rate), "-ac", str(audio.channels)]
            sound += ["-i", str(pcm)]
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "-",
            *sound,
            *(["-c:a", "aac", "-b:a", "192k", "-shortest"] if audio is not None else []),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(out),
        ]
        try:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except FileNotFoundError as e:
            raise EncodeError("ffmpeg is not installed") from e
        with proc:
            stdin, err = proc.stdin, proc.stderr
            assert stdin is not None
            assert err is not None
            stderr: list[bytes] = []
            reader = threading.Thread(target=lambda: stderr.append(err.read()))
            reader.start()
            expected = width * height * 3
            try:
                for frame in frames:
                    if len(frame) != expected:
                        raise EncodeError(f"Frame is {len(frame)} bytes, expected {expected}")
                    stdin.write(frame)
            except BrokenPipeError:
                pass
            except BaseException:
                proc.kill()
                raise
            finally:
                stdin.close()
                code = proc.wait()
                reader.join()
        if code != 0:
            raise EncodeError(f"ffmpeg failed: {b''.join(stderr).decode(errors='replace')[-300:]}")
        return out.read_bytes()
