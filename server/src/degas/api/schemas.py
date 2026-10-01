"""Request bodies."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

Tags = Annotated[list[Annotated[str, Field(min_length=1, max_length=40)]], Field(max_length=20)]


class StartSession(BaseModel):
    gpu: str
    high_mem: bool = False


class SubmitJob(BaseModel):
    spec: dict[str, Any]
    batch_count: int = Field(1, ge=1, le=16)
    seed_mode: Literal["increment", "random"] = "increment"


class MoveJob(BaseModel):
    position: Annotated[int, Field(ge=0)]  # index in the queue; 0 runs next


class PushKeys(BaseModel):
    p256dh: Annotated[str, Field(min_length=1, max_length=200)]
    auth: Annotated[str, Field(min_length=1, max_length=100)]


class PushSubscription(BaseModel):
    endpoint: Annotated[str, Field(pattern=r"^https://", max_length=2000)]
    keys: PushKeys


class PushEndpoint(BaseModel):
    endpoint: str


class FromUrl(BaseModel):
    url: Annotated[str, Field(min_length=1, max_length=20_000_000)]


class CivitaiImport(BaseModel):
    url: Annotated[str, Field(min_length=1, max_length=2000)]
    family: str | None = None
    name: Annotated[str, Field(max_length=120)] | None = None
    weight: Annotated[float, Field(ge=0, le=2)] | None = None
    force: bool = False

    def options(self) -> dict[str, Any]:
        return self.model_dump(exclude={"url"})


class Transform(BaseModel):
    ops: list[dict[str, Any]]


class RemapMask(BaseModel):
    source: str  # the image the mask was painted on
    to: str  # another crop of the same original


class Preprocess(BaseModel):
    id: str
    image: str
    params: dict[str, Any] = {}


class Frame(BaseModel):
    at: Literal["first", "last"] | Annotated[float, Field(ge=0)] = "first"


class LibraryEdit(BaseModel):
    title: Annotated[str, Field(max_length=200)] | None = None
    tags: Tags | None = None


class NewPrompt(BaseModel):
    name: Annotated[str, Field(max_length=200)] = ""
    prompt: str
    negative_prompt: str = ""
    family: str | None = None
    tags: Tags = []


class PromptEdit(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    tags: Tags | None = None
