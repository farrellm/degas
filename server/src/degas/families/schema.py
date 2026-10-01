"""Builders for the parameters every family's schema has (the Create form is rendered from
these: `x-widget` names the control, `x-advanced` puts it under More settings)."""

from degas.families.base import JsonSchema

SEED_LIMIT = 2**32


def prompt_prop() -> JsonSchema:
    return {"type": "string", "title": "Prompt", "minLength": 1, "x-widget": "prompt"}


def negative_prompt_prop(description: str | None = None) -> JsonSchema:
    prop: JsonSchema = {"type": "string", "title": "Negative prompt"}
    if description:
        prop["description"] = description
    return {**prop, "default": "", "x-widget": "prompt"}


def size_props(
    default: tuple[int, int], *, minimum: int, maximum: int, multiple_of: int
) -> dict[str, JsonSchema]:
    """`width` and `height`, which the form shows as one aspect picker."""
    width, height = default
    return {
        name: {
            "type": "integer",
            "title": title,
            "default": value,
            "minimum": minimum,
            "maximum": maximum,
            "multipleOf": multiple_of,
            "x-widget": "aspect",
        }
        for name, title, value in (("width", "Width", width), ("height", "Height", height))
    }


def steps_prop(default: int, maximum: int, description: str | None = None) -> JsonSchema:
    prop: JsonSchema = {"type": "integer", "title": "Steps"}
    if description:
        prop["description"] = description
    return {**prop, "default": default, "minimum": 1, "maximum": maximum, "x-widget": "slider"}


def seed_prop() -> JsonSchema:
    """-1 asks for a random seed."""
    return {
        "type": "integer",
        "title": "Seed",
        "default": -1,
        "minimum": -1,
        "maximum": SEED_LIMIT - 1,
        "x-widget": "seed",
    }


def choice_prop(
    title: str,
    choices: dict[str, str],
    default: str,
    *,
    description: str | None = None,
    advanced: bool = False,
) -> JsonSchema:
    """A select over `choices` (value → label)."""
    prop: JsonSchema = {"type": "string", "title": title}
    if description:
        prop["description"] = description
    prop.update(
        {
            "default": default,
            "enum": list(choices),
            "x-enum-labels": list(choices.values()),
            "x-widget": "select",
        }
    )
    if advanced:
        prop["x-advanced"] = True
    return prop


def params_schema(props: dict[str, JsonSchema]) -> JsonSchema:
    return {"type": "object", "required": ["prompt"], "properties": props}
