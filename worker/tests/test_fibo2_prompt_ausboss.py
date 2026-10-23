import json

from PIL import Image

from degas_worker.families.fibo2_prompt import parse_caption
from degas_worker.families.fibo2_prompt_ausboss import EDIT, T2I, messages, repair


def test_a_new_picture_sends_the_idea_in_the_t2i_prompt() -> None:
    [message] = messages("  a fox in snow\n", None)
    assert message["role"] == "user"
    [part] = message["content"]
    assert part["text"] == T2I.replace("{user}", "a fox in snow")
    assert part["text"].endswith("The idea:\na fox in snow\n")


def test_an_edit_shows_the_picture_first() -> None:
    im = Image.new("RGB", (8, 8))
    [message] = messages("make it snow", im)
    image, text = message["content"]
    assert image == {"type": "image", "image": im}
    assert text["text"] == EDIT.replace("{user}", "make it snow")
    assert "{user}" not in text["text"]


def test_repair_tidies_the_answer_into_json() -> None:
    answer = (
        'Here it is:\n```json\n{"short_description": "a fox",'
        ' "objects": [{"description": "fox",},],\n```\n  '
    )
    assert json.loads(repair(answer)) == {
        "short_description": "a fox",
        "objects": [{"description": "fox"}],
    }
    assert repair('{"a": 1}') == '{"a": 1}'


def test_an_edit_answer_keeps_its_instruction() -> None:
    answer = repair('{"short_description": "snowy street", "edit_instruction": "Add snow."')
    caption = json.loads(parse_caption(answer, "make it snow"))
    assert caption == {"short_description": "snowy street", "edit_instruction": "Add snow."}
