"""The MCP server mounted at `/mcp`: Qwen-Image 2.1's `create_image` and `edit_image`, for
agents. Each call queues a job, starts a GPU session if none is up, and waits for the image."""

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
from degas.errors import DegasError
from degas.services import Services

IMAGES = """Images are numbered in the order given, and the prompt refers to them by \
number ("the jacket from image 2"). Each image only sees the ones before it, so put the most \
important first. An image is a `sha256:…` ref (from an earlier result), an https URL or a \
`data:` URI. Up to 10 images in all."""

INSTRUCTIONS = f"""Degas makes images with Qwen-Image 2.1 on a Colab GPU. `create_image` \
makes a new image from a prompt, optionally drawing on reference images; `edit_image` changes \
an image. A call starts a GPU session if none is running, which takes several minutes; later \
calls take about a minute at 2K. Each result gives its `sha256` ref, to pass to `edit_image` \
or as a reference. {IMAGES}"""

CREATE = f"""Make a new image from a prompt with Qwen-Image 2.1.

With no reference images it draws from the text alone. With references, they are images 1, \
2, … and the prompt says what to take from each; the output's shape comes from aspect_ratio, \
not from the images. {IMAGES}

Describe the picture as a whole: subject, setting, light, medium. Examples:
- (no references) "A lighthouse on a basalt cliff in thick fog at dawn, 35mm photo"
- "The woman from image 1 wearing the leather jacket from image 2, walking down a rainy Tokyo \
street at night"
- "The perfume bottle from image 1 on the marble counter from image 2, soft window light, \
product photo"
- "A portrait of the dog in image 1, painted in the style of image 2"
"""

EDIT = f"""Edit an image with Qwen-Image 2.1, optionally using reference images.

`image` is image 1; the references are images 2, 3, …. The output keeps image 1's shape unless \
aspect_ratio is given (then image 1 is cropped to it). {IMAGES}

Say what to change, and what to keep. Examples:
- "Replace the sky with a stormy sunset; keep everything else unchanged"
- "Put the hat from image 2 on the man in image 1"
- "Change the background to the beach in image 2; keep the person's pose and lighting"
- "Remove the people in the background"
- "Make it a pencil sketch"
- 'Replace the text on the sign with "OPEN LATE"'
"""

Prompt = Annotated[str, Field(min_length=1, description="What to draw, or what to change")]
References = Annotated[
    list[str], Field(description="Reference images, numbered from the first image", max_length=10)
]
Resolution = Annotated[
    generate.Resolution, Field(description="2K is about 4 megapixels, 1K about 1")
]
Seed = Annotated[int | None, Field(description="Fixes the noise, to repeat a result")]
Upsample = Annotated[
    bool,
    Field(
        description="Have Qwen's text encoder first rewrite a short prompt into a full "
        "description (for an edit, a precise instruction from what it sees)"
    ),
]


def build_mcp(get_services: Callable[[], Services]) -> MCPServer:
    mcp = MCPServer("Degas", instructions=INSTRUCTIONS)

    @mcp.tool(description=CREATE, structured_output=False)
    async def create_image(
        ctx: Context,
        prompt: Prompt,
        reference_images: References = [],  # noqa: B006 - pydantic copies the default
        aspect_ratio: generate.Aspect = "1:1",
        resolution: Resolution = "2K",
        seed: Seed = None,
        upsample_prompt: Upsample = False,
        negative_prompt: Annotated[str, Field(description="What to avoid")] = "",
    ) -> list[Any]:
        svc = get_services()
        try:
            size = generate.preset_size(aspect_ratio, resolution)
            images = [
                await generate.image_ref(svc, v, n) for n, v in enumerate(reference_images, 1)
            ]
            # Nothing of image 1 is cut off: the output's shape is the one asked for.
            spec = generate.build_spec(
                svc, prompt, images, size, "pad", seed, upsample_prompt, negative_prompt
            )
            return await _run(ctx, svc, spec)
        except DegasError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(description=EDIT, structured_output=False)
    async def edit_image(
        ctx: Context,
        image: Annotated[str, Field(description="The image to edit (image 1)")],
        prompt: Prompt,
        reference_images: Annotated[
            list[str], Field(description="More images, numbered from 2", max_length=9)
        ] = [],  # noqa: B006 - pydantic copies the default
        aspect_ratio: Annotated[
            generate.Aspect | None, Field(description="Default: image 1's shape")
        ] = None,
        resolution: Resolution = "2K",
        seed: Seed = None,
        upsample_prompt: Upsample = False,
    ) -> list[Any]:
        svc = get_services()
        try:
            images = [
                await generate.image_ref(svc, v, n)
                for n, v in enumerate([image, *reference_images], 1)
            ]
            if aspect_ratio is None:
                w, h = svc.blobs.image_size(unref(images[0])) or (1, 1)
                size = generate.matching_size(w, h, resolution)
            else:
                size = generate.preset_size(aspect_ratio, resolution)
            spec = generate.build_spec(svc, prompt, images, size, "crop", seed, upsample_prompt)
            return await _run(ctx, svc, spec)
        except DegasError as e:
            raise ToolError(str(e)) from e

    return mcp


def security() -> TransportSecuritySettings:
    """The tailnet name reaches the server through `tailscale serve`, so the Host header isn't
    localhost; only the tailnet can reach it, as for the REST API."""
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


async def _run(ctx: Context, svc: Services, spec: dict[str, Any]) -> list[Any]:
    count = 0

    async def report(message: str) -> None:
        nonlocal count
        count += 1
        await ctx.report_progress(count, message=message)

    base = _base_url(ctx)
    out: list[Any] = []
    for made in await generate.run(svc, spec, report):
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
