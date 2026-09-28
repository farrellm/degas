"""MP4 encoding with ffmpeg (preinstalled on Colab). Free of torch so it can be tested anywhere.

H.264 in yuv420p with the index at the front, so it plays inline on iOS (design §4.3).
"""

import subprocess
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path


class EncodeError(RuntimeError):
    pass


def encode_mp4(
    frames: Iterable[bytes], width: int, height: int, fps: float, crf: int = 18
) -> bytes:
    """Encode raw RGB24 frames (`width * height * 3` bytes each)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
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
