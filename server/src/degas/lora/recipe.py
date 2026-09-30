"""The files a run hands to kohya sd-scripts: dataset config, arguments, sample prompts."""

import json
import shlex
from dataclasses import dataclass

from degas.lora.settings import LoraSettings

SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts"
# 0.12.0 pins diffusers 0.40.0 and transformers 5.17, next to Colab's image, and not torch.
SD_SCRIPTS_TAG = "v0.12.0"
LORA_HOME = "/content/lora"  # on the VM: sd-scripts/, bin/, models/, runs/<run>/
STARTED_MARKER = "DEGAS_LORA_STARTED"


@dataclass(frozen=True)
class RemoteRun:
    """Where a run lives on the VM."""

    run: str

    @property
    def dir(self) -> str:
        return f"{LORA_HOME}/runs/{self.run}"

    @property
    def dataset(self) -> str:
        return f"{self.dir}/dataset"

    @property
    def output(self) -> str:
        return f"{self.dir}/output"

    @property
    def samples(self) -> str:
        return f"{self.output}/sample"

    @property
    def log(self) -> str:
        return f"{self.dir}/train.log"

    @property
    def exit_code(self) -> str:
        return f"{self.dir}/exit_code"

    @property
    def script(self) -> str:
        return f"{self.dir}/run.sh"


def model_path(base: str) -> str:
    return f"{LORA_HOME}/models/{base}"


def _toml(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    return repr(value)


def dataset_toml(settings: LoraSettings, remote: RemoteRun, images: int) -> str:
    t = settings.train
    general: dict[str, str | int | float | bool] = {
        "enable_bucket": True,
        "bucket_no_upscale": False,
        "caption_extension": ".txt",
        # Natural-language captions: keep their order. (Caching the text encoder's outputs
        # rules shuffling out anyway.)
        "shuffle_caption": False,
    }
    dataset: dict[str, str | int | float | bool] = {
        "resolution": t.resolution,
        "batch_size": t.batch_size,
        "min_bucket_reso": t.min_bucket_reso,
        "max_bucket_reso": t.max_bucket_reso,
        "bucket_reso_steps": 64,
    }
    subset: dict[str, str | int | float | bool] = {
        "image_dir": remote.dataset,
        "num_repeats": t.repeats_for(images),
    }
    lines = ["[general]", *(f"{k} = {_toml(v)}" for k, v in general.items()), ""]
    lines += ["[[datasets]]", *(f"{k} = {_toml(v)}" for k, v in dataset.items()), ""]
    lines += ["  [[datasets.subsets]]", *(f"  {k} = {_toml(v)}" for k, v in subset.items())]
    return "\n".join(lines) + "\n"


def sample_prompts(settings: LoraSettings) -> str:
    s = settings.samples
    lines = []
    for prompt in s.prompts:
        text = prompt.format(subject=settings.subject, class_word=settings.class_word)
        lines.append(
            f"{text} --n {s.negative} --w {s.width} --h {s.height} --d {s.seed}"
            f" --l {s.cfg:g} --s {s.steps}"
        )
    return "\n".join(lines) + "\n"


def train_args(settings: LoraSettings, remote: RemoteRun) -> list[str]:
    """Arguments for `sdxl_train_network.py`."""
    t = settings.train
    args = [
        f"--pretrained_model_name_or_path={model_path(settings.base)}",
        f"--dataset_config={remote.dir}/dataset.toml",
        f"--output_dir={remote.output}",
        f"--output_name={settings.name}",
        "--save_model_as=safetensors",
        "--save_precision=fp16",
        "--save_every_n_epochs=1",
        f"--max_train_epochs={t.epochs}",
        "--network_module=networks.lora",
        f"--network_dim={t.network_dim}",
        f"--network_alpha={t.network_alpha:g}",
        "--network_train_unet_only",
        f"--optimizer_type={t.optimizer}",
        f"--learning_rate={t.lr:g}",
        f"--lr_scheduler={t.scheduler}",
    ]
    if t.optimizer != "Prodigy" and t.lr_warmup > 0:
        args.append(f"--lr_warmup_steps={t.lr_warmup:g}")  # below 1: a share of the steps
    if t.opt_args:
        args += ["--optimizer_args", *t.opt_args]
    args += [
        f"--mixed_precision={settings.mixed_precision}",
        "--gradient_checkpointing",
        "--sdpa",
        # The checkpoint's own SDXL VAE overflows in half precision; it only encodes the
        # dataset once (latents are cached), so fp32 costs little.
        "--no_half_vae",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        f"--seed={t.seed}",
        "--max_data_loader_n_workers=2",
        "--persistent_data_loader_workers",
    ]
    if t.min_snr_gamma is not None:
        args.append(f"--min_snr_gamma={t.min_snr_gamma:g}")
    if t.noise_offset is not None:
        args.append(f"--noise_offset={t.noise_offset:g}")
    s = settings.samples
    if s.prompts:
        args += [
            f"--sample_prompts={remote.dir}/prompts.txt",
            f"--sample_every_n_epochs={s.every_n_epochs}",
            f"--sample_sampler={s.sampler}",
        ]
        if s.at_first:
            args.append("--sample_at_first")
    return args + t.extra_args


def run_script(settings: LoraSettings, remote: RemoteRun) -> str:
    """`run.sh`: train, then record the exit code (the follower polls for it)."""
    args = " \\\n  ".join(shlex.quote(a) for a in train_args(settings, remote))
    return f"""#!/bin/bash
cd {LORA_HOME}/sd-scripts
python3 -m accelerate.commands.launch --num_processes=1 --num_machines=1 \\
  --mixed_precision={settings.mixed_precision} --dynamo_backend=no \\
  --num_cpu_threads_per_process=1 sdxl_train_network.py \\
  {args}
echo $? > {remote.exit_code}
"""


def launch_code(remote: RemoteRun) -> str:
    """Kernel code that starts `run.sh` detached, so it inherits the CUDA environment
    (Phase 0, finding 6) and survives this `exec` returning."""
    return f"""
import os, subprocess
_env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")
_log = open({remote.log!r}, "ab")
_p = subprocess.Popen(["bash", {remote.script!r}], env=_env, cwd={remote.dir!r},
    stdin=subprocess.DEVNULL, stdout=_log, stderr=subprocess.STDOUT, start_new_session=True)
print({STARTED_MARKER!r}, _p.pid)
"""
