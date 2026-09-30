"""`degas lora`: dataset checks, the sd-scripts recipe, and a run against a fake VM."""

import hashlib
import random
import shlex
import tomllib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from degas.config import Config
from degas.drive import AccessToken
from degas.lora import dataset as ds
from degas.lora import recipe
from degas.lora.cli import find_run
from degas.lora.publish import publish
from degas.lora.run import (
    BusyError,
    RunDir,
    RunError,
    RunState,
    Trainer,
    checkpoint_for,
    format_progress,
    new_run_id,
)
from degas.lora.settings import LoraSettings, load_settings

BASE = "models/sdxl/checkpoint.safetensors"


def settings(**train: Any) -> LoraSettings:
    return LoraSettings(
        name="jane", trigger="ohwx", class_word="woman", base=BASE, train=train or {}
    )


def write_image(path: Path, size: tuple[int, int] = (1024, 1280), seed: int = 0) -> None:
    """Random 8x8 blocks, so average hashes differ between seeds."""
    rng = random.Random(seed)
    im = Image.new("RGB", size)
    bw, bh = size[0] // 8, size[1] // 8
    for x in range(8):
        for y in range(8):
            shade = rng.choice((20, 235))
            im.paste((shade, shade, shade), (x * bw, y * bh, (x + 1) * bw, (y + 1) * bh))
    im.save(path)


def make_dataset(root: Path, n: int = 20) -> Path:
    root.mkdir()
    for i in range(n):
        write_image(root / f"img{i}.png", seed=i)
        (root / f"img{i}.txt").write_text(f"photo of ohwx woman, pose {i}\n")
    return root


# -- settings -------------------------------------------------------------------------------


def test_load_settings_defaults_name_to_the_folder(tmp_path: Path) -> None:
    root = tmp_path / "jane-doe"
    root.mkdir()
    (root / "lora.toml").write_text(
        f'trigger = "ohwx"\nclass_word = "woman"\nbase = "{BASE}"\n[train]\nnetwork_dim = 16\n'
    )
    s = load_settings(root, {"gpu": "A100", "base": None})
    assert (s.name, s.gpu, s.train.network_dim, s.base) == ("jane-doe", "A100", 16, BASE)
    assert s.subject == "ohwx woman"


def test_optimizer_defaults() -> None:
    adam = settings().train
    assert (adam.lr, adam.scheduler, adam.opt_args) == (1e-4, "cosine", [])
    prodigy = settings(optimizer="Prodigy").train
    assert (prodigy.lr, prodigy.scheduler) == (1.0, "constant")
    assert "safeguard_warmup=True" in prodigy.opt_args


def test_repeats_aim_for_the_target_steps() -> None:
    t = settings().train  # 2000 steps, 10 epochs, batch 2
    assert t.repeats_for(20) == 20
    assert t.steps_for(20) == 2000
    assert t.repeats_for(1000) == 1
    assert settings(repeats=3).train.repeats_for(20) == 3


def test_settings_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="letters"):
        LoraSettings(name="a b", trigger="t", class_word="c", base=BASE)
    with pytest.raises(ValueError, match="safetensors"):
        LoraSettings(name="a", trigger="t", class_word="c", base="models/../x.safetensors")
    with pytest.raises(ValueError, match="' --'"):
        LoraSettings(
            name="a", trigger="t", class_word="c", base=BASE, samples={"prompts": ["x --n y"]}
        )


# -- dataset --------------------------------------------------------------------------------


def test_check_a_good_dataset(tmp_path: Path) -> None:
    report = ds.check(make_dataset(tmp_path / "d"), settings())
    assert report.ok, report.errors
    assert len(report.items) == 20
    assert report.warnings == []


def test_check_reports_caption_problems(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d")
    (root / "img0.txt").unlink()
    (root / "img1.txt").write_text("  \n")
    (root / "img2.txt").write_text("photo of a woman")  # no trigger
    (root / "img3.txt").write_text("photo of ohwx")  # no class word
    (root / "stray.txt").write_text("ohwx woman")
    report = ds.check(root, settings())
    assert any("img0.png: no caption" in e for e in report.errors)
    assert any("img1.png: the caption is empty" in e for e in report.errors)
    assert any("img2.png: the caption lacks the trigger" in e for e in report.errors)
    assert any("img3.png: the caption lacks the class word" in w for w in report.warnings)
    assert any("stray.txt" in w for w in report.warnings)


def test_check_warns_about_small_and_duplicate_images(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d")
    write_image(root / "img5.png", size=(512, 512), seed=5)
    write_image(root / "img6.png", seed=4)  # same picture as img4
    warnings = ds.check(root, settings()).warnings
    assert any("img5.png: 512x512 is small" in w for w in warnings)
    assert any("img4.png and img6.png look like duplicates" in w for w in warnings)


def test_check_needs_a_few_images(tmp_path: Path) -> None:
    report = ds.check(make_dataset(tmp_path / "d", n=3), settings())
    assert any("at least 5" in e for e in report.errors)


def test_prepare_applies_exif_rotation_and_flattens_alpha(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "d", n=5)
    im = Image.new("RGB", (1200, 900), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° clockwise to display
    im.save(root / "phone.jpg", exif=exif)
    (root / "phone.txt").write_text("ohwx woman outdoors")
    Image.new("RGBA", (3000, 3000), (0, 0, 0, 0)).save(root / "cutout.png")
    (root / "cutout.txt").write_text("ohwx woman")
    report = ds.check(root, settings())
    assert report.ok
    phone = next(i for i in report.items if i.image.name == "phone.jpg")
    assert phone.size == (900, 1200)

    out = tmp_path / "prepared"
    ds.prepare(report, out)
    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 14  # 7 images + 7 captions
    cutout_idx = [i.image.name for i in report.items].index("cutout.png")
    with Image.open(out / f"{cutout_idx:03d}.png") as im:
        assert im.mode == "RGB"
        assert im.size == (2048, 2048)
        assert im.getpixel((5, 5)) == (255, 255, 255)
    phone_idx = [i.image.name for i in report.items].index("phone.jpg")
    with Image.open(out / f"{phone_idx:03d}.jpg") as im:
        assert im.size == (900, 1200)
    assert (out / f"{phone_idx:03d}.txt").read_text() == "ohwx woman outdoors\n"


# -- recipe ---------------------------------------------------------------------------------


def test_dataset_toml() -> None:
    remote = recipe.RemoteRun("jane-1")
    config = tomllib.loads(recipe.dataset_toml(settings(), remote, images=25))
    assert config["general"]["shuffle_caption"] is False
    assert config["general"]["caption_extension"] == ".txt"
    (dataset,) = config["datasets"]
    assert (dataset["resolution"], dataset["batch_size"]) == (1024, 2)
    (subset,) = dataset["subsets"]
    assert subset == {"image_dir": "/content/lora/runs/jane-1/dataset", "num_repeats": 16}


def test_train_args() -> None:
    remote = recipe.RemoteRun("jane-1")
    args = recipe.train_args(settings(), remote)
    for flag in (
        f"--pretrained_model_name_or_path=/content/lora/models/{BASE}",
        "--network_train_unet_only",
        "--network_dim=32",
        "--network_alpha=16",
        "--optimizer_type=AdamW8bit",
        "--learning_rate=0.0001",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0.05",
        "--mixed_precision=bf16",
        "--cache_text_encoder_outputs",
        "--min_snr_gamma=5",
        "--sample_prompts=/content/lora/runs/jane-1/prompts.txt",
        "--sample_at_first",
    ):
        assert flag in args
    assert "--optimizer_args" not in args

    t4 = LoraSettings(
        name="jane", trigger="ohwx", class_word="woman", base=BASE, gpu="T4",
        train={"optimizer": "Prodigy", "extra_args": ["--noise_offset_random_strength"]},
    )  # fmt: skip
    args = recipe.train_args(t4, remote)
    assert "--mixed_precision=fp16" in args
    assert not any(a.startswith("--lr_warmup_steps") for a in args)
    i = args.index("--optimizer_args")
    assert args[i + 1] == "decouple=True"
    assert args[-1] == "--noise_offset_random_strength"


def test_run_script_quotes_arguments() -> None:
    s = settings(extra_args=["--output_config=it's"])
    script = recipe.run_script(s, recipe.RemoteRun("jane-1"))
    assert shlex.quote("--output_config=it's") in script
    assert "echo $? > /content/lora/runs/jane-1/exit_code" in script


def test_sample_prompts() -> None:
    lines = recipe.sample_prompts(settings()).splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("photo of ohwx woman, head and shoulders")
    assert lines[0].endswith("--w 896 --h 1152 --d 1234 --l 5 --s 28")
    assert lines[5].startswith("photo of a woman,")  # no trigger: checks for bleed


def test_format_progress() -> None:
    tail = (
        "steps:  11%|█         | 220/2000 [04:50<39:00,  1.31s/it, avr_loss=0.101]\n"
        "steps:  12%|█▏        | 240/2000 [05:10<37:50,  1.29s/it, avr_loss=0.0921]\n"
    )
    assert format_progress(tail) == "step 240/2000 · 1.29s/it · 37:50 left · loss 0.0921"
    assert format_progress("caching latents") is None


# -- a run against a fake VM ----------------------------------------------------------------


class FakeColab:
    session = "degas-lora"

    def __init__(self, alive: bool = False) -> None:
        self.alive = alive
        self.calls: list[str] = []

    async def new(self, gpu: str | None, high_mem: bool) -> None:
        self.calls.append(f"new {gpu}")
        self.alive = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self.alive = False

    async def is_alive(self) -> bool:
        return self.alive

    async def exec(self, code: str, timeout: float = 120) -> str:
        self.calls.append("exec pass" if code == "pass" else "exec launch")
        return f"{recipe.STARTED_MARKER} 42\n"

    def proxy_command(self, identity: str) -> str:
        return "true"


class FakeVm:
    """A tunnel whose polls replay a script of (exit code, files, log)."""

    def __init__(self, polls: list[tuple[int | None, dict[str, int], str]]) -> None:
        self.polls = polls
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.downloads: list[str] = []
        self.busy = False

    @property
    def local_port(self) -> int:
        return 1

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def is_open(self) -> bool:
        return True

    async def run(self, command: str, timeout: float = 120) -> str:
        self.commands.append(command)
        if command.startswith("pgrep"):
            return "busy\n" if self.busy else ""
        if command.startswith("echo @@exit"):
            code, files, log = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
            listing = "".join(f"{size} {rel}\n" for rel, size in files.items())
            return f"@@exit\n{'' if code is None else code}\n\n@@files\n{listing}@@log\n{log}\n"
        return ""

    async def upload(self, local: Path, remote: str) -> None:
        self.uploads[remote] = _read(local)

    async def download(self, remote: str, local: Path) -> None:
        self.downloads.append(remote)
        _write(local, b"log line\n" if remote.endswith(".log") else remote.encode())


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class FakeDrive:
    async def access_token(self, min_valid_s: float = 300) -> AccessToken:
        return AccessToken("tok", 2_000_000_000)


def make_run(tmp_path: Path, vm: FakeVm, colab: FakeColab) -> tuple[Trainer, list[str]]:
    s = settings(epochs=2)
    run = new_run_id("jane")
    run_dir = RunDir(tmp_path / "lora-runs" / run)
    report = ds.check(make_dataset(tmp_path / "d", n=6), s)
    ds.prepare(report, run_dir.dataset)
    state = RunState(
        run=run, session="degas-lora", settings=s, images=6, steps=100, created_at="now"
    )
    run_dir.save(state)
    said: list[str] = []
    trainer = Trainer(
        run_dir, state, colab, vm, FakeDrive(), "degas", say=said.append, poll_s=0,
        heartbeat_s=0,
    )  # fmt: skip
    return trainer, said


async def test_a_run_from_start_to_finish(tmp_path: Path) -> None:
    progress = "steps:  50%|█| 50/100 [01:00<01:00,  1.2s/it, avr_loss=0.1]"
    polls: list[tuple[int | None, dict[str, int], str]] = [
        (None, {}, "caching latents"),
        (None, {"jane-000001.safetensors": 10, "sample/jane_e000001_00_1_1234.png": 5}, progress),
        (None, {"jane-000001.safetensors": 10, "sample/jane_e000001_00_1_1234.png": 5}, progress),
        (0, {"jane-000001.safetensors": 10, "jane.safetensors": 10}, progress),
    ]
    vm, colab = FakeVm(polls), FakeColab()
    trainer, said = make_run(tmp_path, vm, colab)
    await trainer.start()
    assert colab.calls == ["new L4", "exec launch"]
    assert trainer.state.status == "training"
    remote = f"/content/lora/runs/{trainer.state.run}"
    assert set(vm.uploads) == {f"{remote}/run.tar", "/content/lora/rclone.conf"}
    assert b'"access_token": "tok"' in vm.uploads["/content/lora/rclone.conf"]
    fetch = next(c for c in vm.commands if "copyto" in c)
    assert f"drive:degas/{BASE}" in fetch
    assert "rm -f /content/lora/rclone.conf" in fetch

    assert await trainer.follow() == "done"
    run_dir = trainer.dir
    assert run_dir.load().status == "done"
    assert (run_dir.checkpoints / "jane-000001.safetensors").exists()
    assert (run_dir.checkpoints / "jane.safetensors").exists()
    assert (run_dir.samples / "jane_e000001_00_1_1234.png").exists()
    assert vm.downloads.count(f"{remote}/output/jane-000001.safetensors") == 1
    assert "step 50/100 · 1.2s/it · 01:00 left · loss 0.1" in said
    assert "exec pass" in colab.calls  # heartbeat
    assert checkpoint_for(run_dir, run_dir.load(), 2).name == "jane.safetensors"
    assert checkpoint_for(run_dir, run_dir.load(), 1).name == "jane-000001.safetensors"
    with pytest.raises(RunError, match="jane-000003"):
        checkpoint_for(run_dir, run_dir.load(), 3)

    await trainer.stop_vm()
    assert colab.calls[-1] == "stop"
    assert run_dir.load().status == "done"  # stopping afterwards doesn't change it


async def test_a_failed_run(tmp_path: Path) -> None:
    trainer, said = make_run(tmp_path, FakeVm([(1, {}, "")]), FakeColab())
    await trainer.start()
    assert await trainer.follow() == "failed"
    state = trainer.dir.load()
    assert (state.status, state.error) == ("failed", "sd-scripts exited with 1")
    assert any("log line" in s for s in said)


async def test_refuses_a_vm_that_is_already_training(tmp_path: Path) -> None:
    vm = FakeVm([(None, {}, "")])
    vm.busy = True
    colab = FakeColab(alive=True)
    trainer, _ = make_run(tmp_path, vm, colab)
    with pytest.raises(BusyError, match="already training"):
        await trainer.start()
    assert colab.calls == []
    assert trainer.dir.load().status == "failed"


def test_find_run_by_name_picks_the_latest(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    for run in ("jane-20260101-000000", "jane-20260201-000000"):
        (tmp_path / "lora-runs" / run).mkdir(parents=True)
        (tmp_path / "lora-runs" / run / "state.json").write_text("{}")
    assert find_run(config, "jane").path.name == "jane-20260201-000000"
    assert find_run(config, "jane-20260101-000000").path.name == "jane-20260101-000000"
    with pytest.raises(RunError):
        find_run(config, "bob")


# -- publish --------------------------------------------------------------------------------


def test_publish(tmp_path: Path) -> None:
    s = settings(epochs=2)
    run_dir = RunDir(tmp_path / "run")
    state = RunState(run="jane-1", session="x", settings=s, images=6, steps=10, created_at="n")
    run_dir.checkpoints.mkdir(parents=True)
    ckpt = run_dir.checkpoints / "jane-000001.safetensors"
    ckpt.write_bytes(b"weights")
    run_dir.samples.mkdir()
    for i in range(2):
        Image.new("RGB", (896, 1152), (i * 100, 0, 0)).save(
            run_dir.samples / f"jane_e000001_{i:02d}_20260101_1234.png"
        )
    calls: list[tuple[str, ...]] = []
    stdins: dict[str, bytes] = {}

    def fake_rclone(*args: str, stdin: bytes | None = None) -> str:
        calls.append(args)
        if stdin is not None:
            stdins[args[1]] = stdin
        if args[0] == "md5sum":
            return f"{hashlib.md5(b'weights').hexdigest()}  jane.safetensors\n"
        if args[0] == "lsf":
            return "other.safetensors\n"
        return ""

    path = publish(
        run_dir, state, remote="gdrive:", drive_root="degas", epoch=1, preview_index=1,
        run_rclone=fake_rclone, say=lambda _: None,
    )  # fmt: skip
    assert path == "loras/sdxl/jane.safetensors"
    folder = "gdrive:degas/loras/sdxl"
    assert calls[0] == ("mkdir", folder)
    assert ("copyto", str(ckpt), f"{folder}/jane.safetensors") in calls
    sidecar = stdins[f"{folder}/jane.yaml"].decode()
    assert "trigger_words:\n- ohwx woman\n" in sidecar
    assert "default_weight: 0.8" in sidecar
    assert "epoch 1/2" in sidecar
    preview = stdins[f"{folder}/jane.jpg"]
    assert preview[:2] == b"\xff\xd8"


def test_publish_refuses_to_replace_without_force(tmp_path: Path) -> None:
    run_dir = RunDir(tmp_path / "run")
    state = RunState(
        run="jane-1", session="x", settings=settings(epochs=1), images=6, steps=10,
        created_at="n",
    )  # fmt: skip
    run_dir.checkpoints.mkdir(parents=True)
    (run_dir.checkpoints / "jane.safetensors").write_bytes(b"w")

    def fake_rclone(*args: str, stdin: bytes | None = None) -> str:
        return "jane.safetensors\n" if args[0] == "lsf" else ""

    with pytest.raises(RunError, match="already exists"):
        publish(
            run_dir, state, remote="gdrive:", drive_root="degas", epoch=None,
            run_rclone=fake_rclone, say=lambda _: None,
        )  # fmt: skip
