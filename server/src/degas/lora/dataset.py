"""Check a LoRA dataset folder, and prepare the copy that is uploaded for training.

A dataset is a folder of images, each with a caption in `<stem>.txt` that contains the
trigger word. sd-scripts ignores EXIF orientation and can't read HEIC, so `prepare` writes
upright PNG/JPEG copies (and scales down anything larger than it will ever train at).
"""

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from degas import media  # noqa: F401 - registers the HEIC opener
from degas.lora.settings import LoraSettings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
MIN_IMAGES = 5
TYPICAL_IMAGES = (15, 60)
SMALL_SIDE = 768  # shorter than this and sd-scripts upscales it into a 1024² bucket
PREPARED_MAX_SIDE = 2048  # the largest bucket side at 1024² with max_bucket_reso 1536
NEAR_DUPLICATE_BITS = 4  # average-hash Hamming distance
CLIP_TOKENS = 75  # sd-scripts' default max_token_length; the rest is cut off


@dataclass
class Item:
    image: Path
    caption: str | None
    size: tuple[int, int]  # upright (after EXIF orientation)
    ahash: int


@dataclass
class Report:
    items: list[Item] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _ahash(im: Image.Image) -> int:
    small = im.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    pixels = list(small.tobytes())  # mode L: one byte per pixel
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, p in enumerate(pixels) if p > mean)


def _approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _word(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)


def _check_caption(report: Report, name: str, caption: str, settings: LoraSettings) -> None:
    if not _word(settings.trigger).search(caption):
        report.errors.append(f"{name}: the caption lacks the trigger {settings.trigger!r}")
    elif not _word(settings.class_word).search(caption):
        report.warnings.append(f"{name}: the caption lacks the class word {settings.class_word!r}")
    if _approx_tokens(caption) > CLIP_TOKENS:
        report.warnings.append(
            f"{name}: the caption is over ~{CLIP_TOKENS} tokens; the rest is ignored"
        )


def _check_image(report: Report, path: Path, settings: LoraSettings) -> None:
    try:
        with Image.open(path) as im:
            upright = ImageOps.exif_transpose(im)
            size = upright.size
            ahash = _ahash(upright)
    except OSError as e:
        report.errors.append(f"{path.name}: can't read it ({e})")
        return
    caption_file = path.with_suffix(".txt")
    caption = caption_file.read_text().strip() if caption_file.exists() else None
    report.items.append(Item(path, caption, size, ahash))
    if caption is None:
        report.errors.append(f"{path.name}: no caption ({caption_file.name})")
    elif not caption:
        report.errors.append(f"{path.name}: the caption is empty")
    else:
        _check_caption(report, path.name, caption, settings)
    if min(size) < SMALL_SIDE:
        report.warnings.append(
            f"{path.name}: {size[0]}x{size[1]} is small; it will be upscaled and look soft"
        )


def check(dataset: Path, settings: LoraSettings) -> Report:
    report = Report()
    if not dataset.is_dir():
        report.errors.append(f"{dataset} is not a folder")
        return report
    images = sorted(p for p in dataset.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    stems: dict[str, Path] = {}
    for path in images:
        if path.stem in stems:
            report.errors.append(f"{path.name} and {stems[path.stem].name} share a caption file")
            continue
        stems[path.stem] = path
        _check_image(report, path, settings)
    for i, a in enumerate(report.items):
        for b in report.items[i + 1 :]:
            if (a.ahash ^ b.ahash).bit_count() <= NEAR_DUPLICATE_BITS:
                report.warnings.append(f"{a.image.name} and {b.image.name} look like duplicates")
    orphans = sorted(p.name for p in dataset.glob("*.txt") if p.stem not in stems)
    if orphans:
        report.warnings.append(f"captions without an image: {', '.join(orphans)}")
    n = len(report.items)
    low, high = TYPICAL_IMAGES
    if n < MIN_IMAGES:
        report.errors.append(f"{n} images: at least {MIN_IMAGES} are needed, 20-40 is typical")
    elif not low <= n <= high:
        report.warnings.append(f"{n} images: 20-40 is typical for a character")
    return report


def prepare(report: Report, dest: Path) -> None:
    """Write `dest/NNN.{png,jpg}` + `NNN.txt`: upright, RGB, at most PREPARED_MAX_SIDE."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for i, item in enumerate(report.items):
        assert item.caption
        stem = f"{i:03d}"
        with Image.open(item.image) as im:
            upright = ImageOps.exif_transpose(im)
            upright.thumbnail((PREPARED_MAX_SIDE, PREPARED_MAX_SIDE), Image.Resampling.LANCZOS)
            has_alpha = upright.mode in ("RGBA", "LA", "PA") or "transparency" in upright.info
            # Transparent areas would train as black: flatten them onto white.
            if has_alpha:
                rgba = upright.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.getchannel("A"))
                upright = flat
            rgb = upright.convert("RGB")
            if item.image.suffix.lower() == ".png":
                rgb.save(dest / f"{stem}.png")
            else:
                rgb.save(dest / f"{stem}.jpg", quality=95)
        (dest / f"{stem}.txt").write_text(item.caption + "\n")
