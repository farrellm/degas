import pytest
from PIL import Image

from degas_worker.families.qwen21_prompt import (
    DEGAS_EDIT,
    DEGAS_T2I,
    SYSTEM,
    clean,
    messages,
)


def test_t2i_sends_the_system_prompt_and_the_prompt() -> None:
    system, user = messages("degas", "  un gatto rosso ", [])
    assert system == {"role": "system", "content": [{"type": "text", "text": DEGAS_T2I}]}
    assert user == {"role": "user", "content": [{"type": "text", "text": "un gatto rosso"}]}


def test_an_edit_shows_one_image_without_a_label() -> None:
    im = Image.new("RGB", (8, 8))
    system, user = messages("kokon", "make it red", [im])
    assert system["content"][0]["text"] == SYSTEM["kokon", "edit"]
    assert user["content"] == [
        {"type": "image", "image": im},
        {"type": "text", "text": "make it red"},
    ]


def test_several_images_are_numbered() -> None:
    a, b = Image.new("RGB", (8, 8)), Image.new("RGB", (4, 4))
    _, user = messages("degas", "put the cat on the sofa", [a, b])
    assert user["content"] == [
        {"type": "text", "text": "<image1>"},
        {"type": "image", "image": a},
        {"type": "text", "text": "<image2>"},
        {"type": "image", "image": b},
        {"type": "text", "text": "put the cat on the sofa"},
    ]


def test_an_unknown_style_is_refused() -> None:
    with pytest.raises(ValueError, match="bria"):
        messages("bria", "x", [])


def test_every_system_prompt_asks_for_the_answer_alone() -> None:
    for (_, kind), text in SYSTEM.items():
        assert text.endswith(
            "Output only the prompt." if kind == "t2i" else "Output only the instruction."
        )
    assert "<image1>, <image2>" in DEGAS_EDIT


def test_degas_prompts_name_no_example_colours() -> None:
    # Kokon found that example colours and objects in a system prompt turn up in its outputs.
    for text in (DEGAS_T2I, DEGAS_EDIT):
        for leak in ("navy", "cream", "copper", "red", "blue", "No text"):
            assert leak not in text


@pytest.mark.parametrize(
    ("answer", "prompt"),
    [
        ("  A watercolour fox.\n", "A watercolour fox."),
        ("Prompt: A fox.", "A fox."),
        ("editing instruction:  Remove the car.", "Remove the car."),
        ('"A fox in snow."', "A fox in snow."),
        ('A sign reading "CAFÉ" above a "BAR".', 'A sign reading "CAFÉ" above a "BAR".'),
        ('"OPEN" glows above a "BAR"', '"OPEN" glows above a "BAR"'),
        ("Ünïcode first", "Ünïcode first"),
        (" \n ", None),
        ("Prompt:", None),
    ],
)
def test_clean(answer: str, prompt: str | None) -> None:
    assert clean(answer) == prompt
