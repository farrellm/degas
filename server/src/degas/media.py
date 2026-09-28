"""Media handling on the server: image normalization, transforms (design §6.5) and ffmpeg.

Pillow work is synchronous (call it from a thread); ffmpeg runs as an asyncio subprocess.
"""

import asyncio
import io
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:  # HEIC from iPhones; optional so the server runs without the plugin
    from pillow_heif import register_heif_opener  # type: ignore[import-not-found,unused-ignore]

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

FIT_MODES = ("crop", "pad", "stretch")
ROTATIONS = (90, 180, 270)  # clockwise
MAX_SIDE = 8192
JPEG_QUALITY = 95

VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}

Op = dict[str, Any]


class MediaError(ValueError):
    pass


# -- images --------------------------------------------------------------------------------


def normalize_image(data: bytes) -> tuple[bytes, str, int, int]:
    """Decode any supported image, apply its EXIF orientation and strip metadata.

    Returns PNG when the image has transparency, else JPEG: `(data, media_type, w, h)`.
    """
    try:
        with Image.open(io.BytesIO(data)) as src:
            im = ImageOps.exif_transpose(src)
            im.load()
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise MediaError("That isn't an image Degas can read") from e
    if max(im.size) > MAX_SIDE:
        raise MediaError(f"The image is larger than {MAX_SIDE} px on a side")
    alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    buf = io.BytesIO()
    if alpha:
        im.convert("RGBA").save(buf, format="PNG")
        return buf.getvalue(), "image/png", im.width, im.height
    im.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), "image/jpeg", im.width, im.height


def validate_ops(ops: Any) -> list[Op]:
    """Check a list of transform operations and return it in canonical form."""
    if not isinstance(ops, list):
        raise MediaError("ops: expected a list")
    out: list[Op] = []
    for op in ops:
        kind = op.get("op") if isinstance(op, dict) else None
        if kind == "rotate":
            if op.get("deg") not in ROTATIONS:
                raise MediaError("rotate: deg must be 90, 180 or 270")
            out.append({"op": "rotate", "deg": op["deg"]})
        elif kind in ("flip_h", "flip_v"):
            out.append({"op": kind})
        elif kind == "crop":
            out.append({"op": "crop", **_ints(op, ("x", "y", "w", "h"))})
        elif kind in ("resize", "pad"):
            dims = _ints(op, ("w", "h"))
            if not (0 < dims["w"] <= MAX_SIDE and 0 < dims["h"] <= MAX_SIDE):
                raise MediaError(f"{kind}: size out of range")
            out.append({"op": kind, **dims, **({"filter": "lanczos"} if kind == "resize" else {})})
        else:
            raise MediaError(f"Unknown transform {kind!r}")
    return out


def _ints(op: dict[str, Any], keys: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        value = op.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MediaError(f"{op.get('op')}: {key} must be a number")
        out[key] = round(value)
    return out


_ROTATE = {
    90: Image.Transpose.ROTATE_270,  # Pillow's names are counter-clockwise
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def apply_ops(data: bytes, ops: list[Op]) -> tuple[bytes, int, int]:
    """Apply operations in order to an image; the result is always PNG (lossless)."""
    try:
        with Image.open(io.BytesIO(data)) as src:
            im = src.convert("RGBA" if "A" in src.getbands() else "RGB")
    except OSError as e:
        raise MediaError("The original image can't be read") from e
    for op in ops:
        match op["op"]:
            case "rotate":
                im = im.transpose(_ROTATE[op["deg"]])
            case "flip_h":
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            case "flip_v":
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            case "crop":
                x, y, w, h = op["x"], op["y"], op["w"], op["h"]
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > im.width or y + h > im.height:
                    raise MediaError(
                        f"crop {w}x{h} at {x},{y} is outside the {im.width}x{im.height} image"
                    )
                im = im.crop((x, y, x + w, y + h))
            case "resize":
                im = im.resize((op["w"], op["h"]), Image.Resampling.LANCZOS)
            case "pad":
                canvas = Image.new(im.mode, (op["w"], op["h"]))
                canvas.paste(im, ((op["w"] - im.width) // 2, (op["h"] - im.height) // 2))
                im = canvas
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), im.width, im.height


def fit_ops(w: int, h: int, tw: int, th: int, mode: str) -> list[Op]:
    """Operations that fit a `w`x`h` image to exactly `tw`x`th` (design §6.5 auto-fit)."""
    if (w, h) == (tw, th):
        return []
    if mode == "stretch":
        return [{"op": "resize", "w": tw, "h": th, "filter": "lanczos"}]
    if mode == "pad":
        scale = min(tw / w, th / h)
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        ops: list[Op] = []
        if (sw, sh) != (w, h):
            ops.append({"op": "resize", "w": sw, "h": sh, "filter": "lanczos"})
        if (sw, sh) != (tw, th):
            ops.append({"op": "pad", "w": tw, "h": th})
        return ops
    # crop: the largest centred rectangle with the target's aspect, then resize
    if w * th > h * tw:  # wider than the target
        cw, ch = max(1, round(h * tw / th)), h
    else:
        cw, ch = w, max(1, round(w * th / tw))
    ops = []
    if (cw, ch) != (w, h):
        ops.append({"op": "crop", "x": (w - cw) // 2, "y": (h - ch) // 2, "w": cw, "h": ch})
    if (cw, ch) != (tw, th):
        ops.append({"op": "resize", "w": tw, "h": th, "filter": "lanczos"})
    return ops


def image_size(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except OSError:
        return None


def thumbnail(image: bytes, size: int) -> bytes:
    with Image.open(io.BytesIO(image)) as im:
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=82)
    return buf.getvalue()


# -- video (ffmpeg) ------------------------------------------------------------------------


async def _run(*args: str, timeout: float = 120) -> bytes:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise MediaError(f"{args[0]} is not installed on the server") from e
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise MediaError(f"{args[0]} timed out") from None
    if proc.returncode != 0:
        tail = err.decode(errors="replace").strip().splitlines()[-3:]
        raise MediaError(f"{args[0]} failed: {' | '.join(tail)}")
    return out


async def probe(path: Path) -> dict[str, Any] | None:
    """Width, height, duration (s), fps and container of a video; None if it isn't one."""
    try:
        out = await _run(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,codec_type:format=duration,format_name",
            "-of",
            "json",
            str(path),
        )
    except MediaError:
        return None
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams or streams[0].get("codec_type") != "video":
        return None
    stream, fmt = streams[0], info.get("format") or {}
    num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    duration = float(fmt.get("duration") or 0)
    if duration <= 0:  # e.g. an image format that ffprobe also reads
        return None
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "format": str(fmt.get("format_name", "")),
    }


def video_media_type(info: dict[str, Any]) -> str:
    fmt = info.get("format", "")
    if "webm" in fmt or "matroska" in fmt:
        return "video/webm"
    return "video/mp4"


async def extract_frame(path: Path, at: str | float) -> bytes:
    """One frame as PNG: `"first"`, `"last"`, or a time in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frame.png"
        if at == "last":
            # Decode the final second, overwriting the file with each frame: the last one stays.
            args = ["-sseof", "-1", "-i", str(path), "-update", "1"]
        else:
            start = 0.0 if at == "first" else max(0.0, float(at))
            args = ["-ss", f"{start:.3f}", "-i", str(path), "-frames:v", "1"]
        await _run("ffmpeg", "-v", "error", "-y", *args, str(out))
        if not out.exists():
            raise MediaError("No frame found at that time")
        return out.read_bytes()


async def stitch(first: Path, second: Path, fps: float) -> bytes:
    """`first` followed by `second` minus its first frame (the frame they share), as H.264."""
    info = await probe(first)
    if info is None:
        raise MediaError("The clip being extended can't be read")
    w, h = info["width"], info["height"]
    rate = fps or info["fps"] or 16
    graph = (
        f"[0:v]fps={rate},setsar=1[a];"
        f"[1:v]fps={rate},trim=start_frame=1,setpts=PTS-STARTPTS,scale={w}:{h},setsar=1[b];"
        "[a][b]concat=n=2:v=1:a=0,format=yuv420p[v]"
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "chain.mp4"
        await _run(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(first),
            "-i",
            str(second),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(out),
            timeout=600,
        )
        return out.read_bytes()
