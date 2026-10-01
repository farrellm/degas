"""A training run's settings: the dataset's `lora.toml`, over these defaults.

The defaults follow the usual SDXL character-LoRA recipe: rank 32 / alpha 16, U-Net only
(kohya's SDXL guide: training both text encoders gives unpredictable results), AdamW8bit at
1e-4 with a cosine schedule, min-SNR gamma 5, ~2000 steps over 10 epochs saved every epoch.
"""

import math
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator

from degas.config import Section

SETTINGS_FILE = "lora.toml"

# `{subject}` is "<trigger> <class_word>"; `{class_word}` alone checks the LoRA doesn't
# bleed into prompts without the trigger. Two prompts are unlike any training image.
DEFAULT_PROMPTS = [
    "photo of {subject}, head and shoulders portrait, looking at the camera, soft window light",
    "photo of {subject}, full body, walking down a city street at night, wearing a red coat",
    "candid photo of {subject} laughing in a busy cafe, 35mm film",
    "photo of {subject} as an astronaut on the surface of the moon, cinematic lighting",
    "impressionist oil painting of {subject} in a garden, loose brushwork",
    "photo of a {class_word}, head and shoulders portrait, looking at the camera,"
    " soft window light",
]
DEFAULT_NEGATIVE = "lowres, blurry, deformed, bad anatomy, extra fingers, watermark, text"


class TrainSettings(Section):
    network_dim: Annotated[int, Field(ge=1, le=256)] = 32
    network_alpha: Annotated[float, Field(gt=0)] = 16
    optimizer: Literal["AdamW8bit", "Prodigy"] = "AdamW8bit"
    learning_rate: float | None = None  # default: 1e-4 (AdamW8bit), 1.0 (Prodigy)
    lr_scheduler: str | None = None  # default: cosine (AdamW8bit), constant (Prodigy)
    lr_warmup: Annotated[float, Field(ge=0, lt=1)] = (
        0.05  # share of the steps; not used with Prodigy
    )
    optimizer_args: list[str] | None = None  # default: Prodigy's usual set, none for AdamW8bit
    epochs: Annotated[int, Field(ge=1)] = 10
    target_steps: Annotated[int, Field(ge=1)] = 2000  # picks the repeats, unless `repeats` is set
    repeats: Annotated[int | None, Field(ge=1)] = None
    batch_size: Annotated[int, Field(ge=1)] = 2
    resolution: int = 1024
    min_bucket_reso: int = 640
    max_bucket_reso: int = 1536
    min_snr_gamma: float | None = 5.0
    noise_offset: float | None = None
    seed: int = 42
    # Passed to sdxl_train_network.py as they are.
    extra_args: list[str] = Field(default_factory=list)

    @property
    def lr(self) -> float:
        if self.learning_rate is not None:
            return self.learning_rate
        return 1.0 if self.optimizer == "Prodigy" else 1e-4

    @property
    def scheduler(self) -> str:
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        return "constant" if self.optimizer == "Prodigy" else "cosine"

    @property
    def opt_args(self) -> list[str]:
        if self.optimizer_args is not None:
            return self.optimizer_args
        if self.optimizer == "Prodigy":
            return [
                "decouple=True",
                "weight_decay=0.01",
                "d_coef=1",
                "use_bias_correction=True",
                "safeguard_warmup=True",
                "betas=0.9,0.99",
            ]
        return []

    def repeats_for(self, images: int) -> int:
        if self.repeats is not None:
            return self.repeats
        return max(1, round(self.target_steps * self.batch_size / (self.epochs * images)))

    def steps_for(self, images: int) -> int:
        """Total optimizer steps (approximate: each aspect bucket rounds up on its own)."""
        return math.ceil(images * self.repeats_for(images) / self.batch_size) * self.epochs


class SampleSettings(Section):
    prompts: list[str] = DEFAULT_PROMPTS
    negative: str = DEFAULT_NEGATIVE
    width: int = 896
    height: int = 1152
    steps: int = 28
    cfg: float = 5.0
    seed: int = 1234
    sampler: str = "dpmsolver++"
    every_n_epochs: int = 1
    at_first: bool = True  # a baseline from the untrained LoRA

    @field_validator("prompts")
    @classmethod
    def _no_option_marker(cls, prompts: list[str]) -> list[str]:
        for p in prompts:
            if " --" in p:  # sd-scripts splits a sample line into options at " --"
                raise ValueError(f"sample prompts can't contain ' --': {p!r}")
        return prompts


class LoraSettings(Section):
    name: str  # the LoRA's file name, without .safetensors
    trigger: str  # a rare token that names the character, e.g. "ohwx"
    class_word: str  # what the character is: "woman", "man", "person"
    base: str  # checkpoint path under the Drive root, e.g. models/sdxl/x.safetensors
    gpu: Literal["T4", "L4", "G4", "A100", "H100"] = "L4"
    high_mem: bool = False
    train: TrainSettings = TrainSettings()
    samples: SampleSettings = SampleSettings()

    @field_validator("name")
    @classmethod
    def _file_name(cls, name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("use letters, digits, '_', '-' and '.' only")
        return name

    @field_validator("base")
    @classmethod
    def _drive_path(cls, base: str) -> str:
        base = base.strip("/")
        if ".." in Path(base).parts or not base.endswith(".safetensors"):
            raise ValueError("expected a .safetensors path under the Drive root")
        return base

    @property
    def subject(self) -> str:
        return f"{self.trigger} {self.class_word}"

    @property
    def mixed_precision(self) -> str:
        return "fp16" if self.gpu == "T4" else "bf16"  # the T4 has no bf16


def load_settings(dataset: Path, overrides: dict[str, Any] | None = None) -> LoraSettings:
    """Read `<dataset>/lora.toml`; `name` defaults to the folder's name."""
    path = dataset / SETTINGS_FILE
    data: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
    data.setdefault("name", dataset.resolve().name)
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return LoraSettings.model_validate(data)
