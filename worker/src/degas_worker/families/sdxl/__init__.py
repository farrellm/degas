"""Stable Diffusion XL runner: text-to-image, image-to-image, inpaint and outpaint, with LoRAs
and ControlNets.

A regular checkpoint is loaded as the text-to-image pipeline, and the image-to-image and
inpaint pipelines are made from it with `from_pipe` (they share its weights and LoRAs). An
inpainting checkpoint (the `inpaint` variant, a 9-channel UNet) only has the inpaint pipeline.
With ControlNet units, the ControlNet version of the mode's pipeline is made the same way.

Regional ControlNet: a unit with an area mask has its ControlNet's residuals multiplied by
the mask, downsampled to each residual's size, so it only guides that area.

Image prompts (IP-Adapter): one adapter per unit, loaded into the shared UNet with the CLIP
image encoder they read pictures with, and unloaded when a job has none. Each unit's scale
names the blocks its purpose uses and is set again between steps for its step range.
"""

from degas_worker.families.sdxl.runner import SdxlRunner

__all__ = ["SdxlRunner"]
