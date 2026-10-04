"""Web Push (design §8.3): VAPID keys, subscriptions and notifications.

The phone gets a notification when a job finishes or fails, when a new session
(or a reset worker) is ready, and when an idle session is about to stop. iOS
delivers them only to a PWA installed to the home screen. Delivery is best effort:
a failure is logged and never affects a job.
"""

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid02
from pywebpush import WebPushException, webpush

from degas.db import Database

log = logging.getLogger(__name__)

# A notification that can't be delivered within this long isn't worth showing.
TTL_S = 3600
SEND_TIMEOUT_S = 15
GONE = (404, 410)  # the push service no longer knows the subscription

# (subscription, JSON payload) → the push service's HTTP status, or None if unreachable.
Sender = Callable[[dict[str, Any], str], Awaitable[int | None]]


def load_vapid(key_file: Path) -> Vapid02:
    """The server's VAPID key pair, generated on first use."""
    if key_file.exists():
        return Vapid02.from_file(str(key_file))
    vapid = Vapid02()
    vapid.generate_keys()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    vapid.save_key(str(key_file))
    key_file.chmod(0o600)
    return vapid


def application_server_key(vapid: Vapid02) -> str:
    """The public key as the browser's `applicationServerKey` (base64url, unpadded)."""
    raw = vapid.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class Push:
    def __init__(
        self, db: Database, key_file: Path, subject: str, send: Sender | None = None
    ) -> None:
        self.db = db
        self.key_file = key_file
        self.subject = subject
        self._vapid: Vapid02 | None = None
        self._send = send or self._webpush
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def vapid(self) -> Vapid02:
        if self._vapid is None:
            self._vapid = load_vapid(self.key_file)
        return self._vapid

    @property
    def public_key(self) -> str:
        return application_server_key(self.vapid)

    def subscribe(self, endpoint: str, keys: dict[str, str]) -> None:
        self.db.add_push_subscription(endpoint, keys)

    def unsubscribe(self, endpoint: str) -> bool:
        return self.db.delete_push_subscription(endpoint)

    def notify(self, title: str, body: str = "", tag: str | None = None, url: str = "/") -> None:
        """Send a notification to every subscribed device, in the background."""
        if not self.db.list_push_subscriptions():
            return
        payload = json.dumps({"title": title, "body": body, "tag": tag, "url": url})
        task = asyncio.create_task(self.send_all(payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def send_all(self, payload: str) -> None:
        for sub in self.db.list_push_subscriptions():
            info = {"endpoint": sub["endpoint"], "keys": sub["keys"]}
            try:
                status = await self._send(info, payload)
            except Exception:
                log.exception("push to %s failed", sub["endpoint"])
                continue
            if status in GONE:
                log.info("push subscription expired: %s", sub["endpoint"])
                self.db.delete_push_subscription(sub["endpoint"])
            elif status is None or status >= 400:
                log.warning("push to %s failed: HTTP %s", sub["endpoint"], status)

    async def drain(self) -> None:
        """Wait for notifications in flight (shutdown and tests)."""
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _webpush(self, subscription: dict[str, Any], payload: str) -> int | None:
        vapid = self.vapid

        def send() -> int | None:
            try:
                response = webpush(
                    subscription,
                    payload,
                    vapid_private_key=vapid,
                    vapid_claims={"sub": self.subject},  # webpush adds aud and exp
                    ttl=TTL_S,
                    timeout=SEND_TIMEOUT_S,
                )
            except WebPushException as e:
                log.debug("push rejected: %s", e)
                return e.response.status_code if e.response is not None else None
            except Exception as e:  # network errors from requests
                log.warning("push service unreachable: %s", e)
                return None
            return int(response.status_code)

        return await asyncio.to_thread(send)
