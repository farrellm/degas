"""Model family registry."""

from degas.families.base import FamilyDescriptor
from degas.families.qwen21 import Qwen21
from degas.families.sdxl import Sdxl
from degas.families.wan22 import Wan22

FAMILIES: dict[str, FamilyDescriptor] = {f.id: f for f in (Sdxl(), Qwen21(), Wan22())}
