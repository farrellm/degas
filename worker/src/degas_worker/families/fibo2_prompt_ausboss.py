# ruff: noqa: E501
"""AusBoss's prompts for Fibo 2.0's caption writer (its "Auto prompt"), verbatim, and how the
answer is tidied into JSON. Torch-free, so it's unit-tested.

From AusBoss's ComfyUI workflow "FIBO 2.0 Generate + Edit | 4-Step Turbo, Auto Prompt" v1.0
(Civitai model 3013987, version 3421012, 2026-10-21; the AusBoss nodes are MIT). There a plain
Qwen3-VL 8B (Qwen-Image 2.1's text encoder, or Qwen3-VL 8B Heretic) gets one of these as the
user turn of its default chat template, with the user's words in place of `{user}` and, for an
edit, the picture. Its answer goes through the workflow's regexes (`repair`), then
`fibo2_prompt.parse_caption`.
"""

import re
from typing import Any

from PIL import Image

T2I = """You write prompts for FIBO 2.0, an image model that reads a JSON description of the picture to draw.

You receive a short idea from the user. Write one JSON object that describes the finished picture in rich detail.

How to write it:
- Keep everything the idea says. Fill in what it leaves open (the place, the light, the colours, the camera angle, the framing, small details) so it fits the idea and makes a strong picture.
- When the idea is about a person, frame them so their face and upper body read clearly (a medium shot or a close-up) unless the idea asks for a wide or full-body view.
- Every person in the picture is an adult. Write "a young woman in her twenties" where the user says girl, and "a young man in his twenties" where the user says boy or guy.
- A photograph unless the idea asks for another style; then set style_medium to that style and describe everything in it.
- Text: put in text_render only words the idea gives, copied letter by letter, and words that belong to an object by nature (STOP on a stop sign). Never invent titles or slogans. Otherwise text_render is [].
- Left and right are each person's own.

The JSON object has exactly these keys:
- "short_description": the picture in a few sentences, 200 words at most.
- "objects": up to five main things, people first. Each has "description", "location" (where in the frame), "relationship" (to the other things), "relative_size", "shape_and_color", "texture", "appearance_details" and "orientation". A person also has "pose", "expression", "clothing", "action", "gender" and "skin_tone_and_texture". A group of the same thing also has "number_of_objects".
- "background_setting": the place and everything in the background.
- "lighting": an object with "conditions", "direction" and "shadows".
- "aesthetics": an object with "composition", "color_scheme" and "mood_atmosphere".
- "photographic_characteristics": an object with "depth_of_field", "focus", "camera_angle" and "lens_focal_length".
- "style_medium": "photograph" unless the idea asks for another style.
- "artistic_style": three words at most.
- "context": what kind of picture this is, in one sentence.
- "text_render": a list of objects with "text", "location", "size", "color", "font" and "appearance_details", or [].

Write only the JSON object, with no words before or after it and no code fence. End with its closing brace.

The idea:
{user}
"""

EDIT = """You write prompts for FIBO 2.0, an image editing model that reads a JSON description of the finished picture.

You receive:
1. A picture.
2. A short edit request from the user.

Write one JSON object that describes the picture as it will look after the edit.

How to write it:
- Look at the picture closely first: each person (face, hair, expression, pose, what each hand does, clothes, jewelry), each object, the place, the light, the colours, the camera angle and the framing.
- Change only what the request asks for, plus what has to change with it (a new place brings its own light; rain makes the street wet). Everything else stays exactly as it is in the picture: the same people with the same faces and hair, the same pose, the same clothes, the same objects in the same places, the same framing, camera angle and lens, the same style.
- Describe the finished picture as it is, never the edit. Write "she wears a white linen shirt", not "her shirt was changed to white". The words changed, added, removed, replaced, now, instead and original belong only in edit_instruction.
- A photograph stays a photograph unless the request asks for another style. When it does, set style_medium to that style (a watercolor painting, an anime illustration, an oil painting) and describe every part of the picture in that style.
- Text: put in text_render the words the request asks to write or change, and large readable words already in the picture that stay (a sign, a print on clothing), copied letter by letter. Otherwise text_render is [].
- Left and right are the person's own: her left hand is on the right of the picture when she faces the camera.

The JSON object has exactly these keys:
- "short_description": the finished picture in a few sentences, 200 words at most.
- "objects": up to five main things, people first. Each has "description", "location" (where in the frame), "relationship" (to the other things), "relative_size", "shape_and_color", "texture", "appearance_details" and "orientation". A person also has "pose", "expression", "clothing", "action", "gender" and "skin_tone_and_texture". A group of the same thing also has "number_of_objects".
- "background_setting": the place and everything in the background.
- "lighting": an object with "conditions", "direction" and "shadows".
- "aesthetics": an object with "composition", "color_scheme" and "mood_atmosphere".
- "photographic_characteristics": an object with "depth_of_field", "focus", "camera_angle" and "lens_focal_length".
- "style_medium": "photograph" unless the request asks for another style.
- "artistic_style": three words at most.
- "context": what kind of picture this is, in one sentence.
- "text_render": a list of objects with "text", "location", "size", "color", "font" and "appearance_details", or [].
- "edit_instruction": the request rewritten as one direct command that starts with a verb (Change, Add, Remove, Make, Turn, Replace, Zoom out) and names what changes the way it looks in the picture. Never mention "the original image".

Write only the JSON object, with no words before or after it and no code fence. End with its closing brace.

The edit request:
{user}
"""

# The workflow's sampling: temperature 0.2, no top-k, top-p or repetition penalty.
TEMPERATURE = 0.2

# The workflow's clean-up, in order: drop code fences, start at the first brace, trim the end,
# close the last brace, drop trailing commas.
REPAIRS = (
    (re.compile(r"```[a-z]*"), ""),
    (re.compile(r"^[^{]*(?=\{)"), ""),
    (re.compile(r"\s+\Z"), ""),
    (re.compile(r"\A(\{[\s\S]*[^}])\Z"), r"\1}"),
    (re.compile(r",(\s*[}\]])"), r"\1"),
)


def messages(prompt: str, image: Image.Image | None) -> list[dict[str, Any]]:
    """The writer's one user turn: the picture of an edit, then its instructions with the
    prompt in them."""
    text = (EDIT if image is not None else T2I).replace("{user}", prompt.strip())
    content: list[dict[str, Any]] = [{"type": "image", "image": image}] if image is not None else []
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def repair(answer: str) -> str:
    """The answer with the slips that would stop it parsing as JSON fixed, as the workflow
    does."""
    for pattern, replacement in REPAIRS:
        answer = pattern.sub(replacement, answer)
    return answer
