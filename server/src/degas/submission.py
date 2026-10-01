"""What a validated spec still needs before it is queued: its assets checked against the
Drive index, and one seed per image."""

import random
from typing import Literal

from degas.db import Database
from degas.families.base import SpecError, lora_files, spec_assets
from degas_worker.spec import Spec

SEED_MAX = 2**32

# Assets a family adds to every job, and what to do when one isn't in Drive.
MISSING = {
    "config": "The pipeline configs aren't in Drive. Put them in degas/{path}/, then rescan.",
    "vae": "The fp16-fix VAE isn't in Drive. Put it in degas/{path}/, then rescan, "
    "or tick Built-in VAE in More settings.",
    "image_encoder": "Image prompts need the CLIP image encoder in Drive. Put it in "
    "degas/{path}/, then rescan.",
    "face_detector": "FaceID needs InsightFace in Drive: det_10g.onnx and w600k_r50.onnx "
    "from buffalo_l in degas/{path}/. Then rescan.",
}

# What the person chose, by the name the form gives it.
CHOSEN = {
    "model": "Model",
    "lora": "LoRA",
    "controlnet": "ControlNet",
    "ip_adapter": "Image prompt model",
}


def resolve_assets(db: Database, family: str, spec: Spec) -> None:
    """Check that every asset the spec names is in the Drive index, and record its size."""
    sizes: dict[str, int | None] = {}
    for need in spec_assets(spec):
        asset = db.get_asset(need["path"])
        owner = None if need["kind"] == "preprocessor" else family  # preprocessors are shared
        if asset is None or asset["kind"] != need["kind"] or asset["family"] != owner:
            key = "face_detector" if need["kind"] == "preprocessor" else need["kind"]
            if key in MISSING:
                raise SpecError(MISSING[key].format(path=need["path"]))
            raise SpecError(f"{CHOSEN[need['kind']]} {need['path']} is not in the Drive index")
        sizes[need["path"]] = asset["size"]
    spec["model"]["size"] = sizes[spec["model"]["path"]]
    for lora in spec["loras"]:
        for part in lora_files(lora):
            part["size"] = sizes[part["path"]]
    for unit in spec.get("control") or []:
        unit["controlnet"]["size"] = sizes[unit["controlnet"]["path"]]
    for prompt in spec.get("image_prompts") or []:
        prompt["adapter"]["size"] = sizes[prompt["adapter"]["path"]]
    added = (
        spec.get("config"),
        spec.get("vae"),
        spec.get("image_encoder"),
        spec.get("face_detector"),
    )
    for extra in added:
        if extra:
            extra["size"] = sizes[extra["path"]]


def batch_seeds(
    seed: int | None, count: int, mode: Literal["increment", "random"], rng: random.Random
) -> list[int]:
    """A batch's seeds: the given one (a random one if it is negative or missing), then either
    the ones after it or more random ones."""
    base = seed if seed is not None and seed >= 0 else rng.randrange(SEED_MAX)
    if mode == "random":
        return [base, *(rng.randrange(SEED_MAX) for _ in range(count - 1))]
    return [(base + i) % SEED_MAX for i in range(count)]
