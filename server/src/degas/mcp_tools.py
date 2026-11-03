"""The MCP server mounted at `/mcp`: `create_image` and `edit_image`, for agents, on the model
`[mcp]` in degas.toml names (Qwen-Image 2.1 by default). Each call queues a job, starts a GPU
session if none is up, and waits for the image."""

import json
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import TextContent
from pydantic import Field

from degas import generate
from degas.blobs import ref, unref
from degas.config import McpConfig
from degas.errors import DegasError
from degas.generate import Target
from degas.services import Services


def _images(target: Target) -> str:
    return f"""Images are numbered in the order given, and the prompt refers to them by \
number ("the jacket from image 2"). Each image only sees the ones before it, so put the most \
important first. An image is a `sha256:…` ref (from an earlier result), an https URL or a \
`data:` URI. Up to {target.max_images} images in all."""


def _instructions(target: Target) -> str:
    tools = (
        "`create_image` makes a new image from a prompt, optionally drawing on reference "
        "images; `edit_image` changes an image."
        if target.can_edit
        else "`create_image` makes a new image from a prompt."
    )
    images = f" {_images(target)}" if target.can_edit else ""
    edit = ", to pass to `edit_image` or as a reference" if target.can_edit else ""
    return f"""Degas makes images with {target.label} on a Colab GPU. {tools} A call starts \
a GPU session ({target.gpu}) if none is running, which takes several minutes; later calls \
take a minute or more, longer with images. Each result gives its `sha256` ref{edit}.{images}"""


def _create(target: Target) -> str:
    if not target.can_edit:
        return f"""Make a new image from a prompt with {target.label}.

Describe the picture as a whole: subject, setting, light, medium. Example: "A lighthouse on a \
basalt cliff in thick fog at dawn, 35mm photo"
"""
    return f"""Make a new image from a prompt with {target.label}.

With no reference images it draws from the text alone. With references, they are images 1, \
2, … and the prompt says what to take from each; the output's shape comes from aspect_ratio, \
not from the images. {_images(target)}

Describe the picture as a whole: subject, setting, light, medium. Examples:
- (no references) "A lighthouse on a basalt cliff in thick fog at dawn, 35mm photo"
- "The woman from image 1 wearing the leather jacket from image 2, walking down a rainy Tokyo \
street at night"
- "The perfume bottle from image 1 on the marble counter from image 2, soft window light, \
product photo"
- "A portrait of the dog in image 1, painted in the style of image 2"
"""


def _edit(target: Target) -> str:
    return f"""Edit an image with {target.label}, optionally using reference images.

`image` is image 1; the references are images 2, 3, …. The output keeps image 1's shape unless \
aspect_ratio is given (then image 1 is cropped to it). {_images(target)}

Say what to change, and what to keep. Examples:
- "Replace the sky with a stormy sunset; keep everything else unchanged"
- "Put the hat from image 2 on the man in image 1"
- "Change the background to the beach in image 2; keep the person's pose and lighting"
- "Remove the people in the background"
- "Make it a pencil sketch"
- 'Replace the text on the sign with "OPEN LATE"'
"""


Prompt = Annotated[str, Field(min_length=1, description="What to draw, or what to change")]
Seed = Annotated[int | None, Field(description="Fixes the noise, to repeat a result")]
Upsample = Annotated[
    bool | None,
    Field(
        description="Have the model's text encoder first rewrite a short prompt into a full "
        "description (for an edit, a precise instruction from what it sees). Default: the "
        "server's setting. Not every model can"
    ),
]
NegativePrompt = Annotated[str, Field(description="What to avoid (not every model takes one)")]


def build_mcp(get_services: Callable[[], Services], config: McpConfig) -> MCPServer:
    target = generate.resolve_target(config)
    mcp = MCPServer("Degas", instructions=_instructions(target))
    res = config.resolution
    with_images = config.resolution_with_images
    resolution_doc = (
        f"2K is about 4 megapixels, 1K about 1. Default: {res} without images, "
        f"{with_images} with them"
    )
    pick = Field(description=resolution_doc)
    # The schema caps how many images a call takes.
    refs = Field(
        description="Reference images, numbered from the first image",
        max_length=target.max_images,
    )

    @mcp.tool(description=_create(target), structured_output=False)
    async def create_image(
        ctx: Context,
        prompt: Prompt,
        reference_images: Annotated[list[str], refs] = [],  # noqa: B006 - pydantic copies the default
        aspect_ratio: generate.Aspect = "1:1",
        resolution: Annotated[generate.Resolution | None, pick] = None,
        seed: Seed = None,
        upsample_prompt: Upsample = None,
        negative_prompt: NegativePrompt = "",
    ) -> list[Any]:
        svc = get_services()
        try:
            images = [
                await generate.image_ref(svc, v, n) for n, v in enumerate(reference_images, 1)
            ]
            resolution = resolution or generate.default_resolution(target, images)
            size = generate.preset_size(target, aspect_ratio, resolution)
            # Nothing of image 1 is cut off: the output's shape is the one asked for.
            spec = generate.build_spec(
                svc, target, prompt, images, size, "pad", seed, upsample_prompt, negative_prompt
            )
            return await _run(ctx, svc, target, spec)
        except DegasError as e:
            raise ToolError(str(e)) from e

    if not target.can_edit:
        return mcp

    @mcp.tool(description=_edit(target), structured_output=False)
    async def edit_image(
        ctx: Context,
        image: Annotated[str, Field(description="The image to edit (image 1)")],
        prompt: Prompt,
        reference_images: Annotated[
            list[str],
            Field(description="More images, numbered from 2", max_length=target.max_images - 1),
        ] = [],  # noqa: B006 - pydantic copies the default
        aspect_ratio: Annotated[
            generate.Aspect | None, Field(description="Default: image 1's shape")
        ] = None,
        resolution: Annotated[generate.Resolution | None, pick] = None,
        seed: Seed = None,
        upsample_prompt: Upsample = None,
    ) -> list[Any]:
        svc = get_services()
        try:
            images = [
                await generate.image_ref(svc, v, n)
                for n, v in enumerate([image, *reference_images], 1)
            ]
            resolution = resolution or generate.default_resolution(target, images)
            if aspect_ratio is None:
                w, h = svc.blobs.image_size(unref(images[0])) or (1, 1)
                size = generate.matching_size(target, w, h, resolution)
            else:
                size = generate.preset_size(target, aspect_ratio, resolution)
            spec = generate.build_spec(
                svc, target, prompt, images, size, "crop", seed, upsample_prompt
            )
            return await _run(ctx, svc, target, spec)
        except DegasError as e:
            raise ToolError(str(e)) from e

    return mcp


def security() -> TransportSecuritySettings:
    """The tailnet name reaches the server through `tailscale serve`, so the Host header isn't
    localhost; only the tailnet can reach it, as for the REST API."""
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


async def _run(ctx: Context, svc: Services, target: Target, spec: dict[str, Any]) -> list[Any]:
    count = 0

    async def report(message: str) -> None:
        nonlocal count
        count += 1
        await ctx.report_progress(count, message=message)

    base = _base_url(ctx)
    out: list[Any] = []
    for made in await generate.run(svc, target, spec, report):
        row = made.result
        info = {
            "sha256": ref(row["blob_sha"]),
            "url": f"{base}/api/blobs/{row['blob_sha']}",
            "width": row["width"],
            "height": row["height"],
            "seed": row["seed"],
            "job_id": row["job_id"],
        }
        if row["caption"]:
            info["prompt_used"] = row["caption"]
        out += [
            Image(data=made.preview, format="jpeg"),
            TextContent(type="text", text=json.dumps(info)),
        ]
    return out


def _base_url(ctx: Context) -> str:
    """Where the client reached the server (uvicorn applies `tailscale serve`'s forwarded
    headers)."""
    url = getattr(ctx.request_context.request, "url", None)
    return f"{url.scheme}://{url.netloc}" if url else ""
