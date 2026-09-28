"""Model family registry."""

from degas.families.base import FamilyDescriptor
from degas.families.sdxl import Sdxl
from degas.families.wan22 import Wan22

FAMILIES: dict[str, FamilyDescriptor] = {f.id: f for f in (Sdxl(), Wan22())}
