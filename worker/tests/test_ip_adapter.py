from typing import Any

import pytest

from degas_worker.families.ip_adapter import (
    active,
    block_scale,
    changed,
    face_loras,
    has_shortcut,
    is_faceid,
    redux_grid,
    scales,
)


def unit(
    purpose: str = "all", weight: float = 0.6, start: float = 0.0, end: float = 1.0
) -> dict[str, Any]:
    return {
        "adapter": {"path": f"ip_adapters/sdxl/{purpose}.safetensors"},
        "purpose": purpose,
        "weight": weight,
        "start": start,
        "end": end,
    }


def test_everything_is_one_weight() -> None:
    assert block_scale("all", 0.6) == 0.6


def test_purposes_name_instantstyle_blocks() -> None:
    assert block_scale("style", 1.0) == {"up": {"block_0": [0.0, 1.0, 0.0]}}
    assert block_scale("layout", 0.5) == {"down": {"block_2": [0.0, 0.5]}}
    assert block_scale("style_layout", 0.8) == {
        "down": {"block_2": [0.0, 0.8]},
        "up": {"block_0": [0.0, 0.8, 0.0]},
    }
    with pytest.raises(ValueError, match="purpose"):
        block_scale("face", 1.0)


def test_steps_follow_the_controlnet_rule() -> None:
    # From 0 to 0.5 of 10 steps: steps 0 to 4.
    assert [active(0.0, 0.5, i, 10) for i in range(10)] == [True] * 5 + [False] * 5
    assert [active(0.2, 1.0, i, 10) for i in range(10)] == [False] * 2 + [True] * 8


def test_scales_are_zero_outside_a_units_steps() -> None:
    units = [unit(end=0.5), unit("style", 1.0, start=0.5)]
    assert scales(units, 0, 4) == [0.6, 0.0]
    assert scales(units, 3, 4) == [0.0, {"up": {"block_0": [0.0, 1.0, 0.0]}}]


def test_scales_are_set_again_only_where_a_unit_starts_or_stops() -> None:
    units = [unit(end=0.5)]
    assert [changed(units, i, 4) for i in range(5)] == [None, None, [0.0], None, None]


def test_faceid_models_are_told_by_name() -> None:
    assert is_faceid("ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin")
    assert not is_faceid("ip_adapters/sdxl/ip-adapter-plus-face_sdxl_vit-h.safetensors")
    assert has_shortcut("ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin")
    assert not has_shortcut("ip_adapters/sdxl/ip-adapter-faceid-plus_sd15.bin")


def test_faceid_loras_are_named_by_their_units_place() -> None:
    faceid = {**unit(), "adapter": {"path": "ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin"}}
    assert face_loras([unit(), {**faceid, "lora_weight": 0.5}]) == [("faceid_1", 0.5)]
    assert face_loras([faceid]) == [("faceid_0", 0.6)]
    assert face_loras([unit()]) == []


def test_redux_grids_shrink_like_comfyui() -> None:
    assert [redux_grid(f) for f in range(1, 6)] == [27, 13, 9, 6, 5]
    with pytest.raises(ValueError, match="1 to 5"):
        redux_grid(6)
