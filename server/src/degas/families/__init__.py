"""Model family registry."""

from degas.families.base import FamilyDescriptor
from degas.families.flux1 import Flux1
from degas.families.klein import Klein
from degas.families.qwen21 import Qwen21
from degas.families.sdxl import Sdxl
from degas.families.wan22 import Wan22

FAMILIES: dict[str, FamilyDescriptor] = {
    f.id: f for f in (Sdxl(), Flux1(), Qwen21(), Klein(), Wan22())
}
