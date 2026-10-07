"""What importing a LoRA means: the family, the Drive files and the sidecar.

A Civitai model version (`plan_import`; a CivArchive one is converted to Civitai's shape first)
or a file in a Hugging Face repo (`plan_hf_import`).

Pure functions over the API's JSON, so they're tested against saved responses.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from degas.blobs import SHA256
from degas.civitai.client import still_url
from degas.civitai.huggingface import MAX_PREVIEW_BYTES, HfRef, quote_path
from degas.errors import DegasError

# Civitai's `baseModel` → (family, the Wan or LTX variants its LoRAs are for). Pony, Illustrious
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
    # Wan 2.1's FLF2V model is fine-tuned from its I2V 720P one, so their LoRAs fit it too.
    "Wan Video 14B i2v 480p": ("wan22", ["wan21-i2v-14b", "wan21-flf2v-14b"]),
    "Wan Video 14B i2v 720p": ("wan22", ["wan21-i2v-14b", "wan21-flf2v-14b"]),
    # LTX-2.0 (Civitai's "LTXV2") is a 19B model, and its LoRAs don't fit the 22B ones.
    "LTXV 2.3": ("ltx2", ["ltx23", "ltx23-distilled"]),
    "LTXV 2.5": ("ltx2", ["ltx25"]),
}
PAIRED_VARIANTS = {"t2v-a14b", "i2v-a14b"}
LORA_TYPES = {"LORA", "LoCon", "DoRA"}
DEFAULT_WEIGHT = 0.8
NAME_MAX = 60
# A Wan A14B file's expert: "…_high_noise", "…-LOW-v2", "high noise" or "LownoiseV2.0" (a
# version's name).
HALF = re.compile(r"(?<![a-z])(high|low)(?=noise|[^a-z]|$)")


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
    from_sibling: bool = False  # the other half, from another version of the model
    mirrors: list[str] = field(default_factory=list)  # to try after `url` (CivArchive)

    @property
    def stem(self) -> str:
        return self.path.rsplit("/", 1)[-1].removesuffix(".safetensors")


@dataclass
class ImportPlan:
    model_id: int | None
    version_id: int | None
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
    paired_version: str | None = None  # the version the other half came from
    origin: str = "civitai"  # or "civarchive", "huggingface"

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
        if self.origin == "huggingface":
            base = f"{self.base_model} " if self.base_model else ""
            data["notes"] = f"{base}LoRA from Hugging Face ({self.version_name})"
        else:
            versions = self.version_name
            if self.paired_version is not None:
                versions += f" + {self.paired_version}"
            site = "CivArchive" if self.origin == "civarchive" else "Civitai"
            data["notes"] = f"{self.base_model} LoRA from {site} ({versions})"
        data["source"] = self.source
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sidecar": self.sidecar()}


def slug(text: str) -> str:
    """A file name from a model's name: "Tést Style XL" → "test_style_xl"."""
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


def pair_key(version_name: str) -> str:
    """A version's name without its half: "highnoiseV2.0" and "LownoiseV2.0" → "v20"."""
    return re.sub(r"[^a-z0-9]|high|low|noise", "", version_name.lower())


def _family(
    base: str, model_name: str, family: str | None, families: set[str]
) -> tuple[str, list[str] | None, list[str]]:
    """(family, Wan or LTX variants, warnings): the base model's, unless `family` overrides it."""
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


def _halves(candidates: list[dict[str, Any]], version_name: str) -> dict[str, dict[str, Any]]:
    """A Wan A14B version's expert files by half ("high", "low")."""
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
    return halves


def _sibling_half(
    siblings: Sequence[dict[str, Any]], version: dict[str, Any], want: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(version, file): the `want` half from another version of the model, when exactly one
    version has only that half, the same base model, and the same name but for high/low
    ("highnoiseV2.0" and "LownoiseV2.0")."""
    key = pair_key(str(version.get("name") or ""))
    found = []
    for other in siblings:
        if other.get("id") == version.get("id") or other.get("baseModel") != version.get(
            "baseModel"
        ):
            continue
        if pair_key(str(other.get("name") or "")) != key:
            continue
        try:
            halves = _halves(_candidates(other), str(other.get("name") or ""))
        except PlanError:
            continue
        if set(halves) == {want}:
            found.append((other, halves[want]))
    return found[0] if len(found) == 1 else None


def _planned(
    f: dict[str, Any],
    version: dict[str, Any],
    path: str,
    half: str | None,
    from_sibling: bool = False,
) -> PlannedFile:
    sha = str((f.get("hashes") or {}).get("SHA256") or "").lower()
    if not SHA256.fullmatch(sha):
        raise PlanError(f"Civitai lists no SHA-256 for {f['name']}")
    url = f.get("downloadUrl") or version.get("downloadUrl")
    if not url:
        raise PlanError(f"No copy of {f['name']} is left to download")
    mirrors = [str(m) for m in f.get("mirrors") or [] if m != url]
    return PlannedFile(
        civitai_name=str(f["name"]),
        url=str(url),
        size=round(float(f.get("sizeKB") or 0) * 1024),
        sha256=sha,
        path=path,
        half=half,
        from_sibling=from_sibling,
        mirrors=mirrors,
    )


def plan_import(
    version: dict[str, Any],
    *,
    families: set[str],
    family: str | None = None,
    name: str | None = None,
    weight: float | None = None,
    siblings: Sequence[dict[str, Any]] = (),
) -> ImportPlan:
    """Plan importing a model version (`GET /api/v1/model-versions/<id>`).

    `family` overrides the one mapped from the base model; `name` is the file name without
    `.safetensors` (for a Wan A14B pair, without `_high_noise` / `_low_noise`). `siblings` are
    the model's versions (`GET /api/v1/models/<id>`): a Wan A14B version with one half takes
    the other from the one that matches it.
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
    paired_version = None
    if variants and PAIRED_VARIANTS & set(variants):
        stem = valid_name(name) if name else slug(model_name) or fallback
        halves = {h: (f, version) for h, f in _halves(candidates, version_name).items()}
        if len(halves) == 1:
            have = next(iter(halves))
            other = "low" if have == "high" else "high"
            if found := _sibling_half(siblings, version, other):
                sibling, f = found
                halves[other] = (f, sibling)
                paired_version = str(sibling.get("name") or "")
            else:
                warnings.append(
                    f"This version has only the {have}-noise half;"
                    f" import the {other}-noise one to complete the pair"
                )
        files = [
            _planned(f, v, f"loras/{family}/{stem}_{h}_noise.safetensors", h, v is not version)
            for h, (f, v) in sorted(halves.items())
        ]
    else:
        named = slug(model_name) and slug(f"{model_name} {version_name}")
        stem = valid_name(name) if name else named or fallback
        files = [_planned(candidates[0], version, f"loras/{family}/{stem}.safetensors", None)]

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
        paired_version=paired_version,
    )


# -- Hugging Face ---------------------------------------------------------------------------

# Hugging Face has no base-model field Degas can rely on: `base_model` in the model card is
# optional, and many LoRAs only say it in their names. These patterns are tried on the file's
# name, then on the card's `base_model`, then on the repo's name: first match wins. (Order matters:
# "ti2v" contains "i2v", and "FLUX.2 klein" contains "flux".)
NAMED_BASES: list[tuple[re.Pattern[str], str, str, list[str] | None]] = [
    (re.compile(p), label, fam, variants)
    for p, label, fam, variants in [
        (r"ltx.?2[._ -]?5", "LTX-2.5", "ltx2", ["ltx25"]),
        (r"ltx.?2[._ -]?3", "LTX-2.3", "ltx2", ["ltx23", "ltx23-distilled"]),
        (r"wan.?2[._ -]?2.*(ti2v|5b)", "Wan 2.2 TI2V-5B", "wan22", ["ti2v-5b"]),
        (r"wan.?2[._ -]?2.*i2v", "Wan 2.2 I2V-A14B", "wan22", ["i2v-a14b"]),
        (r"wan.?2[._ -]?2.*t2v", "Wan 2.2 T2V-A14B", "wan22", ["t2v-a14b"]),
        (
            r"wan.?2[._ -]?1.*(i2v|flf2v)",
            "Wan 2.1 I2V 14B",
            "wan22",
            ["wan21-i2v-14b", "wan21-flf2v-14b"],
        ),
        (r"klein.?9b|9b.?klein", "FLUX.2 klein 9B", "klein", None),
        (r"flux.?1|flux(?![._ -]?2)", "FLUX.1", "flux1", None),
        (r"qwen.?image.?2", "Qwen-Image 2", "qwen21", None),
        (r"sdxl|stable.diffusion.xl|pony|illustrious|noobai", "SDXL", "sdxl", None),
    ]
]
IMAGE = re.compile(r"\.(png|jpe?g|webp)$", re.IGNORECASE)
VIDEO = re.compile(r"\.(mp4|webm|mov)$", re.IGNORECASE)
TRIGGERS = re.compile(
    r"^[\s>*_#-]*(?:trigger(?:s| words?| phrase)?|instance prompt)[*_\s]*:[*_\s]*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _named_base(texts: Sequence[str]) -> tuple[str, str, list[str] | None] | None:
    """(base model's name, family, variants) that `texts` name, the first text first."""
    for text in texts:
        for pattern, label, fam, variants in NAMED_BASES:
            if pattern.search(text.lower()):
                return label, fam, variants
    return None


def _card_bases(card: dict[str, Any]) -> list[str]:
    base = card.get("base_model")
    if isinstance(base, str):
        return [base]
    return [str(b) for b in base] if isinstance(base, list) else []


def readme_triggers(readme: str) -> list[str]:
    """Trigger words from a README line such as "Triggers: 2d animation, Tin"."""
    m = TRIGGERS.search(readme)
    if not m:
        return []
    words = (w.strip(" `*_\"'.") for w in m[1].split(","))
    return [w for w in words if w]


def _hf_family(
    texts: Sequence[str],
    label: str,
    family: str | None,
    hint: str | None,
    families: set[str],
) -> tuple[str, str, list[str] | None, list[str]]:
    """(family, base model's name, variants, warnings) for a Hugging Face LoRA."""
    named = _named_base(texts)
    if family is not None:
        if family not in families:
            raise PlanError(f"Unknown family {family!r}")
        if named is None:
            return family, "", None, []
        base, mapped, variants = named
        if mapped == family:
            return family, base, variants, []
        return family, base, None, [f"It looks like it's for {base}; importing it for {family}"]
    if named is not None:
        base, mapped, variants = named
        return mapped, base, variants, []
    if hint is not None and hint in families:
        warning = f"Its page doesn't say which model it's for; importing it for {hint}"
        return hint, "", None, [warning]
    raise PlanError(f"Can't tell which model {label} is for: choose its family")


def _hf_linked(
    siblings: list[dict[str, Any]], ref: HfRef
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(the LoRA file a link means, the .safetensors files in its folder)."""
    tensors = [f for f in siblings if str(f.get("rfilename", "")).endswith(".safetensors")]
    if ref.path is not None:
        if not ref.path.endswith(".safetensors"):
            raise PlanError(f"{ref.path} isn't a .safetensors file")
        folder = ref.path.rpartition("/")[0]
    else:
        folder = ref.folder.strip("/")
    in_folder = [f for f in tensors if str(f["rfilename"]).rpartition("/")[0] == folder]
    linked = next((f for f in in_folder if ref.path in (None, f["rfilename"])), None)
    if linked is None:
        if ref.path is not None:
            raise PlanError(f"{ref.repo} has no file {ref.path}")
        raise PlanError(f"No .safetensors file in {ref.repo}/{folder}".rstrip("/"))
    return linked, in_folder


def _hf_triggers(card: dict[str, Any], readme: str | None) -> list[str]:
    prompt = card.get("instance_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [prompt.strip()]
    return readme_triggers(readme) if readme else []


def _hf_example(siblings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The repo's first image for a preview, else its smallest video that isn't too big."""
    names = [(f, str(f.get("rfilename", ""))) for f in siblings]
    images = [f for f, n in names if IMAGE.search(n)]
    videos = sorted(
        (f for f, n in names if VIDEO.search(n) and int(f.get("size") or 0) <= MAX_PREVIEW_BYTES),
        key=lambda f: int(f.get("size") or 0),
    )
    found = images or videos
    return found[0] if found else None


def _hf_file(f: dict[str, Any], url: str, path: str, half: str | None) -> PlannedFile:
    name = str(f["rfilename"])
    sha = str((f.get("lfs") or {}).get("sha256") or "").lower()
    if not SHA256.fullmatch(sha):
        raise PlanError(f"Hugging Face lists no SHA-256 for {name}")
    return PlannedFile(
        civitai_name=name.rsplit("/", 1)[-1],
        url=url,
        size=int((f.get("lfs") or {}).get("size") or f.get("size") or 0),
        sha256=sha,
        path=path,
        half=half,
    )


def plan_hf_import(
    info: dict[str, Any],
    ref: HfRef,
    *,
    families: set[str],
    family: str | None = None,
    name: str | None = None,
    weight: float | None = None,
    hint: str | None = None,
    readme: str | None = None,
    base_url: str = "https://huggingface.co",
) -> ImportPlan:
    """Plan importing a LoRA from a Hugging Face repo (`GET /api/models/<repo>?blobs=true`).

    `ref.path` is the file (else the repo, or `ref.folder`, must hold one LoRA, or one Wan
    A14B pair). `hint` is the family to use when nothing names the base model; `readme` is
    the repo's README, for trigger words.
    """
    commit = str(info.get("sha") or ref.revision)
    siblings: list[dict[str, Any]] = info.get("siblings") or []
    linked, in_folder = _hf_linked(siblings, ref)

    def stem(f: dict[str, Any]) -> str:
        return str(f["rfilename"]).rsplit("/", 1)[-1].removesuffix(".safetensors")

    label = stem(linked) if ref.path is not None or len(in_folder) == 1 else ref.repo.split("/")[1]
    card = info.get("cardData") or {}
    # The file's name first: a repo of several LoRAs lists all their bases in its card.
    texts = [str(linked["rfilename"]), *_card_bases(card), ref.repo]
    family, base, variants, warnings = _hf_family(texts, label, family, hint, families)
    fallback = f"hf_{slug(ref.repo)}"

    def url(f: dict[str, Any]) -> str:
        return f"{base_url}/{ref.repo}/resolve/{commit}/{quote_path(str(f['rfilename']))}"

    if variants and PAIRED_VARIANTS & set(variants):
        # The linked file's other half: the file named the same but for high/low.
        key = pair_key(stem(linked))
        pool = [f for f in in_folder if pair_key(stem(f)) == key] if ref.path else in_folder
        halves = _halves([{**f, "name": f"{stem(f)}.safetensors"} for f in pool], "")
        if ref.path is not None and _half(stem(linked)) not in halves:
            raise PlanError(f"Can't tell whether {ref.path} is the high- or low-noise half")
        if len(pool) > len(halves):
            names = ", ".join(stem(f) for f in pool)
            raise PlanError(f"More than one LoRA in {ref.repo}: link to one of {names}")
        label = re.sub(r"[_ -]*(high|low)[_ -]*(noise)?", "", label, flags=re.IGNORECASE) or label
        stem_ = valid_name(name) if name else slug(label) or fallback
        if len(halves) == 1:
            have = next(iter(halves))
            other = "low" if have == "high" else "high"
            warnings.append(
                f"This repo has only the {have}-noise half;"
                f" import the {other}-noise one to complete the pair"
            )
        files = [
            _hf_file(f, url(f), f"loras/{family}/{stem_}_{h}_noise.safetensors", h)
            for h, f in sorted(halves.items())
        ]
    else:
        if ref.path is None and len(in_folder) > 1:
            names = ", ".join(stem(f) for f in in_folder)
            raise PlanError(f"More than one LoRA in {ref.repo}: link to one of {names}")
        stem_ = valid_name(name) if name else slug(label) or fallback
        files = [_hf_file(linked, url(linked), f"loras/{family}/{stem_}.safetensors", None)]

    example = _hf_example(siblings)
    page = f"{base_url}/{ref.repo}"
    if ref.path is not None:
        page += f"/blob/{ref.revision}/{quote_path(ref.path)}"
    return ImportPlan(
        model_id=None,
        version_id=None,
        model_name=label,
        version_name=ref.repo,
        base_model=base,
        family=family,
        source=page,
        label=label,
        trigger_words=_hf_triggers(card, readme),
        weight=DEFAULT_WEIGHT if weight is None else weight,
        variants=variants,
        files=files,
        preview_url=url(example) if example else None,
        warnings=warnings,
        origin="huggingface",
    )
