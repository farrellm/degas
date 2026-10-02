"""LoRA bookkeeping shared by runners: which adapters to load, drop and weight.

Kept free of torch and diffusers so it can be tested without a GPU.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from degas_worker.spec import Lora, PairedLora, Spec


def single_loras(spec: Spec) -> list[Lora]:
    """A spec's LoRAs, for a family whose LoRAs are single files."""
    return cast("list[Lora]", spec.get("loras") or [])


def expert_loras(spec: Spec) -> list[tuple[Lora, str]]:
    """Wan 2.2: each LoRA file with the component it goes into.

    An A14B pair has a file for each expert (`transformer` denoises at high noise,
    `transformer_2` at low); a single file goes into `transformer`.
    """
    files: list[tuple[Lora, str]] = []
    for lora in spec.get("loras") or []:
        if "high" in lora or "low" in lora:
            pair = cast("PairedLora", lora)
            halves = ((pair.get("high"), "transformer"), (pair.get("low"), "transformer_2"))
            files += [(part, component) for part, component in halves if part]
        else:
            files.append((cast("Lora", lora), "transformer"))
    return files


def adapter_name(path: str) -> str:
    """A stable PEFT adapter name for a LoRA file (no dots, which PEFT treats as nesting)."""
    return "lora_" + hashlib.sha1(path.encode(), usedforsecurity=False).hexdigest()[:12]


@dataclass(frozen=True)
class LoraPlan:
    remove: list[str]  # adapter names to delete
    add: list[tuple[str, str]]  # (asset path, adapter name) to load
    names: list[str]  # adapters to activate, in request order
    weights: list[float]


def plan_loras(applied: dict[str, str], requested: list[tuple[str, float]]) -> LoraPlan:
    """Diff the adapters currently loaded (path → name) against the requested (path, weight)."""
    wanted = {path for path, _ in requested}
    remove = [name for path, name in applied.items() if path not in wanted]
    add = [(path, adapter_name(path)) for path, _ in requested if path not in applied]
    names = [applied.get(path) or adapter_name(path) for path, _ in requested]
    return LoraPlan(remove, add, names, [weight for _, weight in requested])


def strip_text_model(keys: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Map `<prefix>.text_model.X` LoRA keys to `<prefix>.X`.

    transformers 5 flattened `CLIPTextModel` (its layers are `encoder.…`, no `text_model.`
    wrapper), but diffusers still converts kohya `lora_te1_*` keys to `text_encoder.text_model.…`,
    so no key matches and PEFT fails with "list index out of range".
    `CLIPTextModelWithProjection` kept the wrapper, so this is only for encoders that lack it.
    """
    old = f"{prefix}.text_model."
    return {
        f"{prefix}.{k.removeprefix(old)}" if k.startswith(old) else k: v for k, v in keys.items()
    }


def qwen21_lora_state(state: dict[str, Any]) -> dict[str, Any]:
    """A Qwen-Image 2.1 LoRA's tensors under names diffusers' loader takes, all of them under
    `diffusion_model.`.

    Two kinds of file load wrongly as they are:

    - Keys under `transformer.` with `.alpha` tensors: the alphas send the file through the
      loader's conversion, which puts a second `transformer.` on and then matches no module.
      It also drops the alphas of `lora_A`/`lora_B` keys, so they are folded into `lora_A` here.
    - The original checkpoint's fused `img_mlp.gate_up`, which the diffusers model has as
      `gate_layer` (its first half of the output rows) and `proj` (the second half): the loader
      skips those keys. Both get the LoRA's down matrix and their half of its up matrix.
    """
    out: dict[str, Any] = {}
    for name, value in state.items():
        prefix = next((p for p in ("transformer.", "diffusion_model.") if name.startswith(p)), "")
        out[name.removeprefix(prefix)] = value
    for key in [k for k in out if k.endswith(".alpha")]:
        down = key.removesuffix("alpha") + "lora_A.weight"
        if down in out:
            out[down] = out[down] * (float(out.pop(key)) / out[down].shape[0])
    for key in [k for k in out if ".gate_up." in k]:
        module, rest = key.split(".gate_up.", 1)
        value = out.pop(key)
        halves = (value, value)
        if rest in ("lora_B.weight", "lora_up.weight"):
            half = value.shape[0] // 2
            halves = (value[:half], value[half:])
        out[f"{module}.gate_layer.{rest}"], out[f"{module}.proj.{rest}"] = halves
    return {f"diffusion_model.{key}": value for key, value in out.items()}
