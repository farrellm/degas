"""Model family registry."""

from degas.families.base import FamilyDescriptor
from degas.families.sdxl import Sdxl

FAMILIES: dict[str, FamilyDescriptor] = {f.id: f for f in (Sdxl(),)}
