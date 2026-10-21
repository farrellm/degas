import json

import pytest
from PIL import Image

from degas_worker.families.fibo2_prompt import (
    NEGATIVE_T2I,
    edit_messages,
    parse_caption,
    t2i_messages,
)


def test_t2i_asks_fibo_vlm_to_generate() -> None:
    [message] = t2i_messages("  a fox in snow ")
    assert message == {
        "role": "user",
        "content": [{"type": "text", "text": "<generate>\na fox in snow"}],
    }


def test_an_edit_shows_the_image_before_the_instruction() -> None:
    im = Image.new("RGB", (8, 8))
    [message] = edit_messages(im, "make it red\n")
    assert message["content"] == [
        {"type": "image", "image": im},
        {"type": "text", "text": "<edit>\nEditing instructions:\nmake it red"},
    ]


def test_the_caption_keeps_known_fields_without_empty_values() -> None:
    answer = json.dumps(
        {
            "short_description": "A fox in snow.",
            "objects": [
                {"description": "a red fox", "pose": "", "colors": []},
                {"description": "", "location": None},
            ],
            "lighting": {"conditions": "overcast", "shadows": float("nan")},
            "aesthetics": {},
            "context": "",
            "aesthetic_score": 9.5,
        }
    )
    caption = parse_caption(answer + "<|im_end|>junk")
    assert caption == (
        '{"short_description":"A fox in snow.","objects":[{"description":"a red fox"}],'
        '"lighting":{"conditions":"overcast"}}'
    )


def test_an_edit_caption_takes_the_reworded_instruction_and_unpacks_nested_json() -> None:
    answer = json.dumps(
        {
            "reworded_prompt": "Make the jacket red.",
            "short_description": "Ünïcode jacket",
            "lighting": json.dumps({"conditions": "soft"}),
        }
    )
    caption = json.loads(parse_caption(answer + " End of texture answer.", "red jacket"))
    assert caption == {
        "short_description": "Ünïcode jacket",
        "lighting": {"conditions": "soft"},
        "edit_instruction": "Make the jacket red.",
    }
    assert "edit_instruction" not in json.loads(parse_caption(answer))


def test_an_edit_caption_without_a_rewording_keeps_the_instruction() -> None:
    caption = json.loads(parse_caption('{"short_description": "x", "reworded_prompt": ""}', "y"))
    assert caption == {"short_description": "x", "edit_instruction": "y"}


@pytest.mark.parametrize("answer", ["not json", "[1, 2]", '{"objects": ["a", "a", "a'])
def test_an_answer_that_isnt_a_json_object_fails(answer: str) -> None:
    with pytest.raises(ValueError, match="prompt VLM"):
        parse_caption(answer)


def test_the_t2i_negative_is_compact_json() -> None:
    negative = json.loads(NEGATIVE_T2I)
    assert negative["aesthetics"]["artifacts"][0] == "oversaturated colours"
    assert " " not in NEGATIVE_T2I.split('"short_description"')[0]
