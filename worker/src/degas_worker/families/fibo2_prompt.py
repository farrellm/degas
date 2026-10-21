"""Fibo 2.0's prompts: what Bria's prompt-to-JSON VLMs are asked, how their answers become the
caption the pipeline reads, and the negatives Bria uses. Torch-free, so it's unit-tested.

Ported from Bria's modular blocks (`briaai/FIBO-VLM-prompt-to-JSON` and
`briaai/FIBO-edit-prompt-to-JSON`, `fibo_vlm_prompt_to_json.py`), without the aesthetic and
preference scores they add, which Fibo 2.0's pipeline drops.
"""

import contextlib
import json
import math
from typing import Any

from PIL import Image

# Where the VLMs' answers end.
STOP = ("<|im_end|>", "<|end_of_text|>")
# The structured caption's fields, in Bria's order.
FIELDS = (
    "short_description",
    "objects",
    "background_setting",
    "lighting",
    "aesthetics",
    "photographic_characteristics",
    "style_medium",
    "text_render",
    "context",
    "artistic_style",
)

# Bria's negative caption for text-to-image (the model card): away from soft, featureless,
# artifact-laden images. An edit uses the pipeline's default, the empty caption.
NEGATIVE_T2I = json.dumps(
    {
        "short_description": "",
        "objects": [
            {
                "description": "the main subject",
                "texture": "smooth, featureless, plastic-like, airbrushed, no fine surface detail",
                "appearance_details": "soft blurred edges, simplified shapes, no fine detail",
                "skin_tone_and_texture": "waxy airbrushed skin, poreless, plastic smooth",
            }
        ],
        "background_setting": "a plain seamless studio backdrop, perfectly smooth gradient, "
        "blurred and featureless, no visible texture",
        "lighting": {},
        "aesthetics": {
            "artifacts": [
                "oversaturated colours",
                "neon colour cast",
                "overexposed highlights",
                "lens flare",
                "slight blur",
                "motion blur",
                "haze",
            ]
        },
        "photographic_characteristics": {
            "depth_of_field": "shallow, heavy bokeh obscuring most of the frame",
            "focus": "soft, slightly out of focus, lacking fine texture",
        },
        "style_medium": "",
        "context": "",
        "artistic_style": "",
    },
    separators=(",", ":"),
)


def t2i_messages(prompt: str) -> list[dict[str, Any]]:
    """FIBO-vlm's request for a caption of a new image."""
    return [
        {"role": "user", "content": [{"type": "text", "text": f"<generate>\n{prompt.strip()}"}]}
    ]


def edit_messages(image: Image.Image, instruction: str) -> list[dict[str, Any]]:
    """FIBO-edit-vlm's request for an edit of `image`."""
    text = f"<edit>\nEditing instructions:\n{instruction.strip()}"
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
        }
    ]


def parse_caption(answer: str, instruction: str | None = None) -> str:
    """The caption in a VLM's answer, as compact JSON: its known fields with the empty values
    (None, "", [], {}, NaN) dropped at every level. For an edit, pass the `instruction` asked
    for: FIBO-edit-vlm writes its rewording as `reworded_prompt` (Fibo Edit 1's name), which
    becomes the `edit_instruction` Fibo 2.0 reads, or the instruction itself if it wrote none.
    Raises ValueError if the answer isn't a JSON object, as when the VLM fell into a loop
    and ran out of tokens."""
    for stop in STOP:
        answer = answer.split(stop, 1)[0]
    answer = answer.replace(" End of texture answer.", "").strip()
    try:
        record = json.loads(answer)
    except json.JSONDecodeError as e:
        raise ValueError(f"The prompt VLM didn't answer with JSON: {answer[:200]!r}") from e
    if not isinstance(record, dict):
        raise ValueError(f"The prompt VLM didn't answer with a JSON object: {answer[:200]!r}")
    edit = instruction is not None
    caption: dict[str, Any] = {}
    for name in FIELDS:
        value = record.get(name)
        if edit and isinstance(value, str):
            # The edit VLM sometimes writes a nested field as a JSON string.
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        value = _prune(value)
        if not _empty(value):
            caption[name] = value
    if instruction is not None:
        reworded = record.get("edit_instruction") or record.get("reworded_prompt")
        caption["edit_instruction"] = (
            reworded.strip() if isinstance(reworded, str) and reworded.strip() else instruction
        )
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if not _empty(v)}
    if isinstance(value, list):
        return [v for v in (_prune(v) for v in value) if not _empty(v)]
    return value


def _empty(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, (str, list, dict)) and not value)
        or (isinstance(value, float) and math.isnan(value))
    )
