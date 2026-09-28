from PIL import Image

from degas_worker.families.control import control_kwargs, crop_areas

UNIT = {"scale": 0.7, "start": 0.0, "end": 0.8}


def test_one_unit_takes_plain_values() -> None:
    image = Image.new("RGB", (8, 8))
    kwargs = control_kwargs([UNIT], [image], "t2i")
    assert kwargs == {
        "image": image,
        "controlnet_conditioning_scale": 0.7,
        "control_guidance_start": [0.0],
        "control_guidance_end": [0.8],
    }


def test_several_units_take_lists() -> None:
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    kwargs = control_kwargs([UNIT, {"scale": 1, "start": 0.2, "end": 1}], images, "inpaint")
    assert kwargs["control_image"] == images  # `image` is the inpaint source
    assert kwargs["controlnet_conditioning_scale"] == [0.7, 1.0]
    assert kwargs["control_guidance_start"] == [0.0, 0.2]
    assert kwargs["control_guidance_end"] == [0.8, 1.0]


def test_areas_are_cropped_like_the_image() -> None:
    area = Image.new("L", (100, 100), 0)
    area.paste(255, (50, 50, 100, 100))
    # The crop takes the middle quarter and scales it back up to the redraw size.
    cropped = crop_areas([area, None], (25, 25, 75, 75), (100, 100))
    assert cropped[1] is None
    first = cropped[0]
    assert first is not None
    assert first.size == (100, 100)
    left, top, right, bottom = first.getbbox() or (0, 0, 0, 0)
    assert (right, bottom) == (100, 100)
    assert 48 <= left <= 50
    assert 48 <= top <= 50
