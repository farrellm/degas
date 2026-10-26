"""Krea 2 edits: fitting and placing the images, and the prompt that sees them."""

import pytest

from degas_worker.families.krea2_edit import (
    PREFIX,
    SUFFIX,
    VISION,
    Fit,
    fit,
    grounded_prompt,
    grounding_size,
    place,
    placements,
    ref_boosts,
)


@pytest.mark.parametrize(
    ("src", "out", "expected"),
    [
        # The output's own shape fills it.
        ((1024, 1024), (1024, 1024), Fit((0, 0, 1024, 1024), (1024, 1024))),
        # Near its aspect ratio, cropped to it and scaled to fill it.
        ((1000, 1050), (1024, 1024), Fit((0, 25, 1000, 1025), (1024, 1024))),
        # A portrait in a landscape output keeps its shape, at the output's height.
        ((600, 1200), (1216, 832), Fit((0, 0, 600, 1200), (416, 832))),
        # Floored to 16 (341 → 336 rows), and cropped first so nothing is squashed.
        ((3000, 1000), (1024, 1024), Fit((0, 8, 3000, 992), (1024, 336))),
        ((2000, 1000), (1024, 1024), Fit((0, 0, 2000, 1000), (1024, 512))),
    ],
)
def test_fit(src: tuple[int, int], out: tuple[int, int], expected: Fit) -> None:
    got = fit(*src, *out)
    assert got == expected
    assert got.size[0] <= out[0]
    assert got.size[1] <= out[1]
    assert got.size[0] % 16 == got.size[1] % 16 == 0


def test_a_fit_s_grid_is_in_tokens() -> None:
    assert fit(600, 1200, 1216, 832).grid == (52, 26)


def test_aligned_images_are_centred_on_the_target() -> None:
    assert place([(64, 64)], (64, 64), ["aligned"]) == [(0.0, 0.0)]
    # Half-token offsets are kept: RoPE positions are continuous.
    assert place([(52, 25)], (52, 76), ["aligned"]) == [(0.0, 25.5)]


def test_beside_goes_right_of_the_target_and_of_earlier_ones() -> None:
    assert place([(52, 25), (30, 20)], (52, 76), ["beside", "beside"]) == [
        (0.0, 76.0),
        (11.0, 101.0),
    ]
    # Identity's scene stays aligned while the person goes beside.
    assert place([(52, 76), (64, 64)], (52, 76), ["aligned", "beside"]) == [
        (0.0, 0.0),
        (0.0, 76.0),
    ]


def test_placement_and_boosts_follow_the_edit_lora() -> None:
    assert placements("general", 1) == ["aligned"]
    assert placements("identity", 1) == ["aligned"]
    assert placements("identity", 2) == ["aligned", "beside"]
    assert ref_boosts("general", 1) == [4.0]
    assert ref_boosts("identity", 1) == [2.0]
    assert ref_boosts("identity", 2) == [1.0, 2.0]


def test_grounded_prompt() -> None:
    assert grounded_prompt("make it night", 1) == PREFIX + VISION + "make it night" + SUFFIX
    assert grounded_prompt("", 2) == PREFIX + VISION + VISION + SUFFIX
    assert PREFIX.startswith("<|im_start|>system\nDescribe the image by detailing the color")
    assert PREFIX.endswith("<|im_end|>\n<|im_start|>user\n")
    assert SUFFIX == "<|im_end|>\n<|im_start|>assistant\n"


def test_grounding_size_caps_the_long_side() -> None:
    assert grounding_size(1536, 1024) == (768, 512)
    assert grounding_size(500, 700) == (500, 700)
