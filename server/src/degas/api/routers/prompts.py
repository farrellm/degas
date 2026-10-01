"""Saved prompts."""

from typing import Any

from fastapi import APIRouter, HTTPException

from degas.api.deps import Svc
from degas.api.schemas import NewPrompt, PromptEdit
from degas.library import clean_tags, prompt_name

router = APIRouter(tags=["prompts"])


@router.get("/prompts")
async def list_prompts(svc: Svc, q: str | None = None) -> list[dict[str, Any]]:
    return svc.db.list_prompts(q)


@router.post("/prompts", status_code=201)
async def save_prompt(svc: Svc, body: NewPrompt) -> dict[str, Any]:
    if not body.prompt.strip():
        raise HTTPException(400, "The prompt is empty")
    saved = svc.db.insert_prompt(
        body.name.strip() or prompt_name(body.prompt),
        body.prompt,
        body.negative_prompt,
        body.family,
        clean_tags(body.tags),
    )
    svc.bus.publish({"type": "prompts"})
    return saved


@router.patch("/prompts/{prompt_id}")
async def edit_prompt(svc: Svc, prompt_id: str, body: PromptEdit) -> dict[str, Any]:
    if svc.db.get_prompt(prompt_id) is None:
        raise HTTPException(404, "Unknown prompt")
    fields: dict[str, Any] = {}
    if body.name is not None and body.name.strip():
        fields["name"] = body.name.strip()
    if body.tags is not None:
        fields["tags"] = clean_tags(body.tags)
    svc.db.update_prompt(prompt_id, **fields)
    svc.bus.publish({"type": "prompts"})
    saved = svc.db.get_prompt(prompt_id)
    assert saved is not None
    return saved


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(svc: Svc, prompt_id: str) -> dict[str, Any]:
    if not svc.db.delete_prompt(prompt_id):
        raise HTTPException(404, "Unknown prompt")
    svc.bus.publish({"type": "prompts"})
    return {"deleted": True}
