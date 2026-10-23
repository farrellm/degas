# ruff: noqa: E501
"""Qwen-Image 2.1's prompt upsampling: what its own text encoder, Qwen3-VL 8B, is asked, and how
its answer becomes the prompt. Torch-free, so it's unit-tested.

After Kokon's "Prompt+" (https://github.com/GabrieleLozupone/kokon): a system prompt for a new
image and one for an edit, Qwen3-VL's recommended sampling, and the answer's clean-up. The
`kokon` style sends Kokon's own system prompts (`qwen21_prompt_kokon.py`); `degas` sends ours,
written from the rules Kokon's notes give. Neither names an example colour or object: Kokon
found the model copies them into every prompt.
"""

from typing import Any

from PIL import Image

from degas_worker.families import qwen21_prompt_kokon as kokon

DEGAS_T2I = """You turn a user's request into a prompt for an image generation model. Write one English paragraph describing the finished image: its subject, setting, composition, light and style.
- Keep every detail the user gave (things, how many, colours, places in the frame, style) and never go against it. A detailed request keeps all its details; only add what it leaves open.
- Numbers are exact for each kind of thing the user lists.
- Carry over the meaning of a request in another language, not its literal words.
- State the medium and style early on. If the user gives none, pick what suits the request, and a photograph for an everyday scene.
- Describe the light every time: where it comes from, its direction and how hard or soft it is.
- Be specific about colours, materials and where each thing sits in the frame; never fall back on vague words.
- For people, describe only what can be seen: age range, expression, pose, hair, clothes. Use no brand names the user didn't give.
- Text the user wants in the image goes in double quotes exactly as given, in its own language and script, with where it is and what it looks like. Otherwise say nothing about text.
- No size, resolution or quality words, and no tags like masterpiece: describe what is visible.
- Write in English, in the present tense, as a description and not as instructions: about 80 to 150 words, or up to about 250 for a scene with many people or things, with a sentence for each.
Output only the prompt."""

DEGAS_EDIT = """You turn a user's request into an instruction for an image editing model. You see the input image and the request. Write one precise English editing instruction.
- Begin with the action (Change, Replace, Remove, Add, Turn, Place, ...), then say what changes, where it is in the image and what it becomes. The change is complete and plain to see.
- Describe only what changes, in detail. Don't describe what stays; end with one sentence such as "Keep everything else in the image unchanged."
- People keep their faces and identities, and the image keeps its medium, unless the user asks otherwise.
- Ground every detail in what the image shows. Invent nothing, and do only what the user asked.
- When something is removed or moved, its shadow and reflection go too; say briefly what fills the space.
- New text in the image goes in double quotes exactly as the user gave it, keeping the existing lettering's style and place unless asked otherwise.
- A change of style, season, time of day or mood applies to the whole image; end with "Keep the composition and the subjects unchanged." instead.
- With several input images, call them <image1>, <image2>, ...; begin with Place or Combine, and say what comes from each image and where it goes.
- Write in English, as one paragraph with no size or aspect ratio: about 20 to 60 words for one change, or a sentence for each of several changes, up to about 150 words.
Output only the instruction."""

# By style, then kind of request. Keep the styles in sync with the server descriptor
# (degas/families/qwen21.py).
SYSTEM = {
    ("kokon", "t2i"): kokon.T2I,
    ("kokon", "edit"): kokon.EDIT,
    ("degas", "t2i"): DEGAS_T2I,
    ("degas", "edit"): DEGAS_EDIT,
}

# Qwen's recommended sampling for Qwen3-VL, as Kokon uses it. The presence penalty counts only
# the tokens generated so far.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5
# The prompts ask for up to about 250 words; the cap only stops a model that doesn't end.
MAX_NEW_TOKENS = 1024
# The encoder sees each image at most this big.
IMAGE_MAX_SIDE = 1024

# Labels the model sometimes puts before its answer.
LABELS = ("Prompt:", "Instruction:", "Editing instruction:")


def messages(style: str, prompt: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    """The chat for the encoder: the style's system prompt, then the images (labelled
    `<imageN>` when there are several) and the user's prompt. With images, it's an edit."""
    system = SYSTEM.get((style, "edit" if images else "t2i"))
    if system is None:
        raise ValueError(f"Unknown upsample style {style!r}")
    content: list[dict[str, Any]] = []
    for n, image in enumerate(images, 1):
        if len(images) > 1:
            content.append({"type": "text", "text": f"<image{n}>"})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt.strip()})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def clean(answer: str) -> str | None:
    """The model's answer as a prompt: without a leading label or quotes around the whole of it.
    None if nothing is left."""
    text = answer.strip()
    for label in LABELS:
        if text[: len(label)].lower() == label.lower():
            text = text[len(label) :].lstrip()
    # Quotes around everything, but not quoted text inside (text to draw in the image).
    if len(text) >= 2 and text[0] == text[-1] == '"' and '"' not in text[1:-1]:
        text = text[1:-1].strip()
    return text or None
