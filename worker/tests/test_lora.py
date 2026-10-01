from typing import Any

from degas_worker.families.lora import (
    adapter_name,
    expert_loras,
    plan_loras,
    single_loras,
    strip_text_model,
)


def test_first_application_loads_everything() -> None:
    plan = plan_loras({}, [("a", 0.8), ("b", 1.0)])
    assert plan.remove == []
    assert plan.add == [("a", adapter_name("a")), ("b", adapter_name("b"))]
    assert plan.names == [adapter_name("a"), adapter_name("b")]
    assert plan.weights == [0.8, 1.0]


def test_only_the_difference_is_loaded_and_removed() -> None:
    applied = {"a": adapter_name("a"), "b": adapter_name("b")}
    plan = plan_loras(applied, [("b", 0.5), ("c", 1.0)])
    assert plan.remove == [adapter_name("a")]
    assert plan.add == [("c", adapter_name("c"))]
    assert plan.names == [adapter_name("b"), adapter_name("c")]
    assert plan.weights == [0.5, 1.0]


def test_no_loras_removes_all() -> None:
    plan = plan_loras({"a": "lora_x"}, [])
    assert (plan.remove, plan.add, plan.names) == (["lora_x"], [], [])


def test_adapter_names_are_stable_and_flat() -> None:
    name = adapter_name("loras/sdxl/film.v3.safetensors")
    assert name == adapter_name("loras/sdxl/film.v3.safetensors")
    assert "." not in name
    assert "/" not in name


def test_strip_text_model_only_touches_its_prefix() -> None:
    keys = {
        "text_encoder.text_model.encoder.layers.0.mlp.fc1.lora_linear_layer.down.weight": 1,
        "text_encoder_2.text_model.encoder.layers.0.mlp.fc1.lora_linear_layer.down.weight": 2,
        "unet.mid_block.attentions.0.proj_in.lora.down.weight": 3,
    }
    assert strip_text_model(keys, "text_encoder") == {
        "text_encoder.encoder.layers.0.mlp.fc1.lora_linear_layer.down.weight": 1,
        "text_encoder_2.text_model.encoder.layers.0.mlp.fc1.lora_linear_layer.down.weight": 2,
        "unet.mid_block.attentions.0.proj_in.lora.down.weight": 3,
    }


def test_single_loras_are_the_specs_list() -> None:
    lora = {"path": "loras/sdxl/a.safetensors", "weight": 0.8, "size": 10}
    spec: Any = {"loras": [lora]}
    none: Any = {}
    assert single_loras(spec) == [lora]
    assert single_loras(none) == []


def test_expert_loras_send_each_half_to_its_expert() -> None:
    high = {"path": "loras/wan22/a_high_noise.safetensors", "weight": 1.0, "size": 1}
    low = {"path": "loras/wan22/a_low_noise.safetensors", "weight": 0.9, "size": 2}
    only_low = {"path": "loras/wan22/b_low_noise.safetensors", "weight": 0.5, "size": 3}
    single = {"path": "loras/wan22/c.safetensors", "weight": 0.7, "size": 4}
    spec: Any = {"loras": [{"high": high, "low": low}, {"low": only_low}, single]}
    assert expert_loras(spec) == [
        (high, "transformer"),
        (low, "transformer_2"),
        (only_low, "transformer_2"),
        (single, "transformer"),
    ]
