"""What importing a Civitai model version means: the family, the Drive files and the sidecar.

Pure functions over the API's JSON, so they're tested against saved responses.
"""

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from degas.blobs import SHA256
from degas.civitai.client import still_url
from degas.errors import DegasError

# Civitai's `baseModel` → (family, the Wan variants its LoRAs are for). Pony, Illustrious
# and NoobAI are SDXL fine-tunes: their LoRAs load on any SDXL checkpoint but look right only on
# their own base, which the sidecar's notes record. Pony V7 is AuraFlow, so it isn't here.
BASE_MODELS: dict[str, tuple[str, list[str] | None]] = {
    "SDXL 1.0": ("sdxl", None),
    "SDXL 0.9": ("sdxl", None),
    "SDXL Turbo": ("sdxl", None),
    "SDXL Lightning": ("sdxl", None),
    "SDXL Hyper": ("sdxl", None),
    "Pony": ("sdxl", None),
    "Illustrious": ("sdxl", None),
    "NoobAI": ("sdxl", None),
    "Flux.1 D": ("flux1", None),
    "Flux.1 S": ("flux1", None),
    "Flux.1 Krea": ("flux1", None),
    "Flux.2 Klein 9B": ("klein", None),
    "Flux.2 Klein 9B-base": ("klein", None),
    "Qwen 2": ("qwen21", None),
    "Qwen 2.1": ("qwen21", None),
    "Wan Video 2.2 TI2V-5B": ("wan22", ["ti2v-5b"]),
    "Wan Video 2.2 T2V-A14B": ("wan22", ["t2v-a14b"]),
    "Wan Video 2.2 I2V-A14B": ("wan22", ["i2v-a14b"]),
    "Wan Video 14B i2v 480p": ("wan22", ["wan21-i2v-14b"]),
    "Wan Video 14B i2v 720p": ("wan22", ["wan21-i2v-14b"]),
}
PAIRED_VARIANTS = {"t2v-a14b", "i2v-a14b"}
LORA_TYPES = {"LORA", "LoCon", "DoRA"}
DEFAULT_WEIGHT = 0.8
NAME_MAX = 60
# A Wan A14B file's expert: "…_high_noise", "…-LOW-v2", "high noise" (a version's name).
HALF = re.compile(r"(?:^|[^a-z])(high|low)(?:[^a-z]|$)")


class PlanError(DegasError, ValueError):
    pass


@dataclass
class PlannedFile:
    civitai_name: str
    url: str
    size: int
    sha256: str  # lowercase hex
    path: str  # under the Drive root: loras/<family>/<name>.safetensors
    half: str | None = None  # "high" or "low" for a Wan A14B expert

    @property
    def stem(self) -> str:
        return self.path.rsplit("/", 1)[-1].removesuffix(".safetensors")


@dataclass
class ImportPlan:
    model_id: int
    version_id: int
    model_name: str
    version_name: str
    base_model: str
    family: str
    source: str
    label: str
    trigger_words: list[str]
    weight: float
    variants: list[str] | None
    files: list[PlannedFile]
    preview_url: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def folder(self) -> str:
        return f"loras/{self.family}"

    def sidecar(self) -> str:
        data: dict[str, Any] = {"label": self.label}
        if self.trigger_words:
            data["trigger_words"] = self.trigger_words
        data["default_weight"] = self.weight
        if self.variants:
            data["variants"] = self.variants
        data["notes"] = f"{self.base_model} LoRA from Civitai ({self.version_name})"
        data["source"] = self.source
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sidecar": self.sidecar()}


def slug(text: str) -> str:
    """A file name from a model's name: "Détail Tweaker XL" → "detail_tweaker_xl"."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9.]+", "_", ascii_).strip("_.").lower()
    return s[:NAME_MAX].rstrip("_.")


def valid_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or len(name) > NAME_MAX * 2:
        raise PlanError(f"{name!r}: use letters, digits, '_', '-' and '.' only")
    return name


def _half(text: str) -> str | None:
    found = {m[1] for m in HALF.finditer(text.lower().replace("_", " "))}
    return found.pop() if len(found) == 1 else None


def _family(
    base: str, model_name: str, family: str | None, families: set[str]
) -> tuple[str, list[str] | None, list[str]]:
    """(family, Wan variants, warnings): the base model's, unless `family` overrides it."""
    mapped, variants = BASE_MODELS.get(base, (None, None))
    if family is None:
        if mapped is None:
            raise PlanError(f"{model_name} is for {base}, which Degas doesn't run")
        return mapped, variants, []
    if family not in families:
        raise PlanError(f"Unknown family {family!r}")
    if family == mapped:
        return family, variants, []
    return family, None, [f"It's for {base}; importing it for {family} anyway"]


def _candidates(version: dict[str, Any]) -> list[dict[str, Any]]:
    """The version's .safetensors model files, the primary one first."""
    files = version.get("files") or []
    found = [
        f
        for f in files
        if f.get("type") == "Model" and str(f.get("name", "")).lower().endswith(".safetensors")
    ]
    if not found:
        names = ", ".join(str(f.get("name")) for f in files) or "none"
        raise PlanError(f"No .safetensors LoRA file in this version (files: {names})")
    for f in found:
        if "Danger" in (f.get("pickleScanResult"), f.get("virusScanResult")):
            raise PlanError(f"Civitai's scan flagged {f['name']}")
    return sorted(found, key=lambda f: not f.get("primary"))


def _halves(
    candidates: list[dict[str, Any]], version_name: str, stem: str, warnings: list[str]
) -> list[tuple[dict[str, Any], str, str | None]]:
    """A Wan A14B version's expert files, named as a pair: `<stem>_high_noise`, …"""
    halves: dict[str, dict[str, Any]] = {}
    for f in candidates:
        half = _half(str(f["name"]).rsplit(".", 1)[0])
        if half is None and len(candidates) == 1:
            half = _half(version_name)
        if half is not None:
            halves.setdefault(half, f)
    if not halves:
        raise PlanError(
            f"Can't tell whether {candidates[0]['name']} is the high- or low-noise half"
        )
    if len(halves) == 1:
        have = next(iter(halves))
        other = "low" if have == "high" else "high"
        warnings.append(
            f"This version has only the {have}-noise half;"
            f" import the {other}-noise one to complete the pair"
        )
    return [(f, f"{stem}_{h}_noise", h) for h, f in sorted(halves.items())]


def _planned(
    f: dict[str, Any], version: dict[str, Any], path: str, half: str | None
) -> PlannedFile:
    sha = str((f.get("hashes") or {}).get("SHA256") or "").lower()
    if not SHA256.fullmatch(sha):
        raise PlanError(f"Civitai lists no SHA-256 for {f['name']}")
    return PlannedFile(
        civitai_name=str(f["name"]),
        url=str(f.get("downloadUrl") or version.get("downloadUrl")),
        size=round(float(f.get("sizeKB") or 0) * 1024),
        sha256=sha,
        path=path,
        half=half,
    )


def plan_import(
    version: dict[str, Any],
    *,
    families: set[str],
    family: str | None = None,
    name: str | None = None,
    weight: float | None = None,
) -> ImportPlan:
    """Plan importing a model version (`GET /api/v1/model-versions/<id>`).

    `family` overrides the one mapped from the base model; `name` is the file name without
    `.safetensors` (for a Wan A14B pair, without `_high_noise` / `_low_noise`).
    """
    model = version.get("model") or {}
    model_name = str(model.get("name") or f"Model {version.get('modelId')}")
    version_name = str(version.get("name") or "")
    base = str(version.get("baseModel") or "unknown")
    if model.get("type") not in LORA_TYPES:
        raise PlanError(f"{model_name} is a {model.get('type', 'model')}, not a LoRA")
    family, variants, warnings = _family(base, model_name, family, families)
    candidates = _candidates(version)

    fallback = f"civitai_{version['id']}"
    if variants and PAIRED_VARIANTS & set(variants):
        stem = valid_name(name) if name else slug(model_name) or fallback
        chosen = _halves(candidates, version_name, stem, warnings)
    else:
        named = slug(model_name) and slug(f"{model_name} {version_name}")
        stem = valid_name(name) if name else named or fallback
        chosen = [(candidates[0], stem, None)]
    files = [
        _planned(f, version, f"loras/{family}/{file_stem}.safetensors", half)
        for f, file_stem, half in chosen
    ]

    words = [w.strip() for w in version.get("trainedWords") or [] if str(w).strip()]
    images = version.get("images") or []
    return ImportPlan(
        model_id=int(version["modelId"]),
        version_id=int(version["id"]),
        model_name=model_name,
        version_name=version_name,
        base_model=base,
        family=family,
        source=f"https://civitai.com/models/{version['modelId']}?modelVersionId={version['id']}",
        label=model_name,
        trigger_words=words,
        weight=DEFAULT_WEIGHT if weight is None else weight,
        variants=variants,
        files=files,
        preview_url=still_url(str(images[0]["url"])) if images else None,
        warnings=warnings,
    )
