"""FLUX.1 [dev] runner: text-to-image (design §4.3).

A single-file checkpoint is only the transformer: it's loaded with `from_single_file` and the
rest of the pipeline (T5, CLIP, VAE, scheduler) comes from the FLUX.1-dev diffusers folder the
spec names as its `config`. A checkpoint stored in fp8 (`flux1-dev-fp8`) stays fp8 on the GPU
through layerwise casting: each layer is cast up to bf16 as it runs, so a 12 GB transformer
doesn't become 24 GB, and casting fp8 weights up and back loses nothing.
"""

import contextlib
import gc
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import plan_loras, strip_text_model
from degas_worker.families.offload import OFFLOAD_ABOVE, place
from degas_worker.safetensors_info import stored_float8

FLOAT8 = {"F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


class Flux1Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.float8 = False  # the transformer is stored in fp8
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        if spec.get("mode", "t2i") != "t2i":
            raise ValueError(f"FLUX.1 can't do {spec['mode']!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        base = None
        if spec.get("config"):
            base = ctx.fetch_asset(spec["config"]["path"], spec["config"].get("size"))
        loras = [
            (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
            for lora in spec.get("loras") or []
        ]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path, base)
        self._apply_loras(loras)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        kwargs: dict[str, Any] = {
            "prompt": params["prompt"],
            "width": int(params["width"]),
            "height": int(params["height"]),
            "num_inference_steps": steps,
            "guidance_scale": float(params["guidance"]),
        }
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)

            def on_step(
                _pipe: Any, i: int, _t: Any, kw: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = i + 1
                ctx.progress(item, "denoise" if done < steps else "decode", done, steps)
                return kw

            result = self.pipe(
                **kwargs,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
            ctx.progress(item, "encode", steps, steps)
            buf = io.BytesIO()
            result.images[0].save(buf, format="PNG")
            yield Output(
                item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png"
            )

    def _load(self, path: Path, base: Path | None) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        storage = None
        if base is None:
            pipe = FluxPipeline.from_pretrained(
                str(path), torch_dtype=torch.bfloat16, local_files_only=True
            )
        else:
            stored = stored_float8(path)
            storage = FLOAT8[stored] if stored else None
            transformer = FluxTransformer2DModel.from_single_file(
                str(path),
                config=str(base),
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe = FluxPipeline.from_pretrained(
                str(base),
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        if storage is None:
            # A bf16 transformer (24 GB) that can't sit on the GPU even by itself, as on an L4,
            # is stored in fp8 there too. That one loses a little precision.
            _free, total = torch.cuda.mem_get_info()
            size = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters())
            if size > total * OFFLOAD_ABOVE:
                storage = torch.float8_e4m3fn
        if storage is not None:
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=storage, compute_dtype=torch.bfloat16
            )
        place(pipe)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.float8 = storage is not None
        self.adapters = {}

    def _apply_loras(self, loras: list[tuple[str, Path, float]]) -> None:
        """Load only new adapters, delete ones no longer requested, then set the weights."""
        plan = plan_loras(self.adapters, [(path, weight) for path, _, weight in loras])
        if plan.remove:
            self.pipe.delete_adapters(plan.remove)
            self.adapters = {p: n for p, n in self.adapters.items() if n not in plan.remove}
        local = {path: file for path, file, _ in loras}
        for path, name in plan.add:
            try:
                with _adapters_in_bf16(self.float8):
                    self._load_lora(local[path], name)
            except Exception as e:
                with contextlib.suppress(Exception):
                    self.pipe.delete_adapters([name])
                raise ValueError(f"Could not load LoRA {path}: {e}") from e
            self.adapters[path] = name
        if plan.names:
            self.pipe.set_adapters(plan.names, adapter_weights=plan.weights)

    def _load_lora(self, file: Path, name: str) -> None:
        """`load_lora_weights`, but with the CLIP keys matched to transformers 5's flattened
        `CLIPTextModel` (see `strip_text_model`). Flux Control LoRAs aren't supported."""
        pipe = self.pipe
        state, alphas, metadata = pipe.lora_state_dict(
            str(file), return_alphas=True, return_lora_metadata=True
        )
        if not hasattr(pipe.text_encoder, "text_model"):
            state = strip_text_model(state, "text_encoder")
            alphas = alphas and strip_text_model(alphas, "text_encoder")
        common = {"network_alphas": alphas, "adapter_name": name, "metadata": metadata}
        pipe.load_lora_into_transformer(
            state, transformer=pipe.transformer, _pipeline=pipe, **common
        )
        pipe.load_lora_into_text_encoder(
            state,
            text_encoder=pipe.text_encoder,
            prefix="text_encoder",
            lora_scale=pipe.lora_scale,
            _pipeline=pipe,
            **common,
        )

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.adapters = {}
        gc.collect()
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _adapters_in_bf16(active: bool) -> Iterator[None]:
    """PEFT moves a new adapter to its base layer's dtype, which for fp8-stored weights is fp8:
    the LoRA's weights would then be copied in at 3 mantissa bits, and fp8 matmuls fail. Move
    it back to bf16 right after, before the weights are loaded. (Layerwise casting skips LoRA
    layers, so they then stay bf16.)"""
    if not active:
        yield
        return
    from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: PLC0415 - only with LoRAs

    original = BaseTunerLayer._move_adapter_to_device_of_base_layer
    fp8 = tuple(FLOAT8.values())

    def patched(self: Any, adapter_name: str, *args: Any, **kwargs: Any) -> None:
        original(self, adapter_name, *args, **kwargs)
        for layer_name in (*self.adapter_layer_names, *self.other_param_names):
            layer = getattr(self, layer_name, None)
            if not isinstance(layer, torch.nn.ModuleDict | torch.nn.ParameterDict):
                continue
            if adapter_name not in layer:
                continue
            part = layer[adapter_name]
            if isinstance(part, torch.nn.Parameter):
                if part.dtype in fp8:
                    layer[adapter_name] = torch.nn.Parameter(
                        part.to(torch.bfloat16), requires_grad=part.requires_grad
                    )
            elif any(p.dtype in fp8 for p in part.parameters()):
                part.to(torch.bfloat16)

    BaseTunerLayer._move_adapter_to_device_of_base_layer = patched
    try:
        yield
    finally:
        BaseTunerLayer._move_adapter_to_device_of_base_layer = original
