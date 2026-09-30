"""Image prompt (IP-Adapter) bookkeeping for runners: which UNet blocks each purpose acts in,
and each unit's scale at each step.

Kept free of torch and diffusers so it can be tested without a GPU.
"""

from typing import Any

# InstantStyle (Wang et al. 2024): in SDXL's UNet one attention block carries the layout
# (the second in down block 2) and one the style (the second in up block 0). Scales use
# diffusers' `set_ip_adapter_scale` form; blocks left out get 0.
_LAYOUT_BLOCKS = ("down", "block_2", [0.0, 1.0])
_STYLE_BLOCKS = ("up", "block_0", [0.0, 1.0, 0.0])
PURPOSE_BLOCKS: dict[str, tuple[tuple[str, str, list[float]], ...]] = {
    "layout": (_LAYOUT_BLOCKS,),
    "style": (_STYLE_BLOCKS,),
    "style_layout": (_LAYOUT_BLOCKS, _STYLE_BLOCKS),
}

Scale = float | dict[str, Any]


def block_scale(purpose: str, weight: float) -> Scale:
    """A unit's scale at full strength: its weight in the blocks its purpose uses."""
    if purpose == "all":
        return weight
    try:
        blocks = PURPOSE_BLOCKS[purpose]
    except KeyError:
        raise ValueError(f"Unknown image prompt purpose {purpose!r}") from None
    scale: dict[str, Any] = {}
    for part, block, ones in blocks:
        scale.setdefault(part, {})[block] = [weight * one for one in ones]
    return scale


def active(start: float, end: float, step: int, steps: int) -> bool:
    """Whether a unit acts on step `step` (0-based) of `steps`, by the rule diffusers uses for
    ControlNet: step / steps ≥ start and (step + 1) / steps ≤ end."""
    return step / steps >= start and (step + 1) / steps <= end


def scales(units: list[dict[str, Any]], step: int, steps: int) -> list[Scale]:
    """Every unit's scale for one step, in unit order (0 outside its steps)."""
    return [
        block_scale(u["purpose"], float(u["weight"]))
        if active(float(u["start"]), float(u["end"]), step, steps)
        else 0.0
        for u in units
    ]


def changed(units: list[dict[str, Any]], step: int, steps: int) -> list[Scale] | None:
    """The scales to set before step `step` (0-based) when a unit starts or stops there;
    None when nothing changes."""
    if step <= 0 or step >= steps:
        return None
    after = scales(units, step, steps)
    return after if after != scales(units, step - 1, steps) else None
