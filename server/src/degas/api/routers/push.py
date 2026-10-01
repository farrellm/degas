"""Web Push subscriptions."""

from typing import Any

from fastapi import APIRouter

from degas.api.deps import Svc
from degas.api.schemas import PushEndpoint, PushSubscription

router = APIRouter(tags=["push"])


@router.get("/push")
async def push_key(svc: Svc) -> dict[str, Any]:
    return {"public_key": svc.push.public_key}


@router.post("/push/subscribe", status_code=201)
async def push_subscribe(svc: Svc, body: PushSubscription) -> dict[str, Any]:
    svc.push.subscribe(body.endpoint, body.keys.model_dump())
    return {"subscribed": True}


@router.post("/push/unsubscribe")
async def push_unsubscribe(svc: Svc, body: PushEndpoint) -> dict[str, Any]:
    return {"unsubscribed": svc.push.unsubscribe(body.endpoint)}
