import json
import struct
from pathlib import Path

from degas_worker.safetensors_info import stored_float8, tensor_dtypes


def checkpoint(path: Path, tensors: dict[str, str]) -> Path:
    """A `.safetensors` file with this header (name → dtype); the data isn't read."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, dtype in tensors.items():
        header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0")
    return path


def test_reads_the_dtypes_without_the_metadata(tmp_path: Path) -> None:
    file = checkpoint(tmp_path / "a.safetensors", {"x.weight": "BF16", "y.weight": "F8_E4M3"})
    assert tensor_dtypes(file) == {"x.weight": "BF16", "y.weight": "F8_E4M3"}


def test_an_fp8_transformer_is_fp8_whatever_its_norms_are(tmp_path: Path) -> None:
    file = checkpoint(
        tmp_path / "flux1-dev-fp8.safetensors",
        {
            "double_blocks.0.img_attn.qkv.weight": "F8_E4M3",
            "double_blocks.0.img_attn.norm.query_norm.scale": "BF16",
            "single_blocks.0.linear1.weight": "F8_E4M3",
            "img_in.weight": "BF16",
            "time_in.in_layer.weight": "BF16",
        },
    )
    assert stored_float8(file) == "F8_E4M3"


def test_all_in_one_files_count_the_transformer_blocks(tmp_path: Path) -> None:
    # ComfyUI's all-in-one checkpoint: a fp8 transformer next to a bf16 CLIP and VAE.
    file = checkpoint(
        tmp_path / "aio.safetensors",
        {
            "model.diffusion_model.double_blocks.0.img_mlp.0.weight": "F8_E5M2",
            "text_encoders.clip_l.transformer.text_model.encoder.layers.0.mlp.fc1.weight": "BF16",
            "vae.decoder.up.0.block.0.conv1.weight": "BF16",
            "vae.decoder.up.0.block.1.conv1.weight": "BF16",
        },
    )
    assert stored_float8(file) == "F8_E5M2"


def test_a_bf16_checkpoint_isnt_fp8(tmp_path: Path) -> None:
    file = checkpoint(
        tmp_path / "flux1-dev.safetensors",
        {"transformer_blocks.0.attn.to_q.weight": "BF16", "x_embedder.weight": "BF16"},
    )
    assert stored_float8(file) is None
