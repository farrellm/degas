"""Google Drive: OAuth (Degas's own desktop client) and the asset index.

The refresh token stays on the server. Short-lived access tokens are minted from
it for the Drive API (indexing) and pushed to the worker (rclone copies).
"""

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx2

SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_API = "https://www.googleapis.com/drive/v3"
FOLDER = "application/vnd.google-apps.folder"
FILE_FIELDS = "id, name, mimeType, size, modifiedTime, md5Checksum"

# Top-level folders under the Drive root, and the asset kind of what they hold.
KINDS = {
    "models": "model",
    "loras": "lora",
    "controlnets": "controlnet",
    "vae": "vae",
    "preprocessors": "preprocessor",
}
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")


class DriveError(RuntimeError):
    pass


class DriveAuthError(DriveError):
    """The refresh token is missing or has been revoked: run `degas auth drive`."""


@dataclass(frozen=True)
class AccessToken:
    token: str
    expires_at: float  # epoch seconds

    @property
    def expiry_rfc3339(self) -> str:
        return datetime.fromtimestamp(self.expires_at, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class OAuthClient:
    client_id: str
    client_secret: str
    auth_uri: str
    token_uri: str

    @classmethod
    def load(cls, path: Path) -> "OAuthClient":
        data = json.loads(path.read_text())
        c = data.get("installed") or data.get("web") or data
        return cls(
            client_id=c["client_id"],
            client_secret=c.get("client_secret", ""),
            auth_uri=c.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
            token_uri=c.get("token_uri", "https://oauth2.googleapis.com/token"),
        )


class DriveAuth:
    def __init__(
        self,
        client_file: Path | None,
        token_file: Path,
        http: httpx2.AsyncClient | None = None,
    ) -> None:
        self.client_file = client_file
        self.token_file = token_file
        self._http = http or httpx2.AsyncClient(timeout=30)
        self._cached: AccessToken | None = None
        self.error: str | None = None  # last refresh failure, for the UI

    @property
    def configured(self) -> bool:
        return self.client_file is not None and self.client_file.exists()

    @property
    def authorized(self) -> bool:
        return self.configured and self.token_file.exists()

    def status(self) -> dict[str, Any]:
        return {"configured": self.configured, "authorized": self.authorized, "error": self.error}

    def _client(self) -> OAuthClient:
        if self.client_file is None or not self.client_file.exists():
            raise DriveAuthError("No Drive OAuth client configured (drive.client_file)")
        return OAuthClient.load(self.client_file)

    async def access_token(self, min_valid_s: float = 300) -> AccessToken:
        if self._cached and self._cached.expires_at - time.time() > min_valid_s:
            return self._cached
        client = self._client()
        if not self.token_file.exists():
            raise DriveAuthError("Drive is not authorized: run `degas auth drive`")
        refresh = json.loads(self.token_file.read_text())["refresh_token"]
        resp = await self._http.post(
            client.token_uri,
            data={
                "client_id": client.client_id,
                "client_secret": client.client_secret,
                "refresh_token": refresh,
                "grant_type": "refresh_token",
            },
        )
        if resp.status_code in (400, 401):
            self.error = f"Drive authorization failed ({resp.text.strip()[:200]})"
            raise DriveAuthError(self.error + ": run `degas auth drive`")
        resp.raise_for_status()
        body = resp.json()
        self._cached = AccessToken(body["access_token"], time.time() + int(body["expires_in"]))
        self.error = None
        return self._cached

    async def aclose(self) -> None:
        await self._http.aclose()


def save_refresh_token(token_file: Path, refresh_token: str) -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"refresh_token": refresh_token}, f)


def authorize_interactive(
    client_file: Path, token_file: Path, port: int = 0, prompt: Callable[[str], None] = print
) -> None:
    """Run the loopback OAuth consent flow and store the refresh token.

    Open the printed URL in a browser on this machine (or forward the port over SSH).
    """
    client = OAuthClient.load(client_file)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    state = secrets.token_urlsafe(16)
    received: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
            if "code" not in query and "error" not in query:
                self.send_response(404)
                self.end_headers()
                return
            received.update(query)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Degas: authorization received. You can close this tab.")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        redirect = f"http://127.0.0.1:{server.server_port}/"
        url = (
            client.auth_uri
            + "?"
            + urllib.parse.urlencode(
                {
                    "client_id": client.client_id,
                    "redirect_uri": redirect,
                    "response_type": "code",
                    "scope": SCOPE,
                    "access_type": "offline",
                    "prompt": "consent",
                    "state": state,
                    "code_challenge": challenge.decode(),
                    "code_challenge_method": "S256",
                }
            )
        )
        prompt(f"Open this URL to authorize Degas to read Google Drive:\n\n{url}\n")
        prompt(f"Waiting for the redirect on {redirect} …")
        while not received:
            server.handle_request()

    if received.get("state") != state:
        raise DriveAuthError("OAuth state mismatch")
    if "error" in received:
        raise DriveAuthError(f"Authorization denied: {received['error']}")
    resp = httpx2.post(
        client.token_uri,
        data={
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "code": received["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    refresh = resp.json().get("refresh_token")
    if not refresh:
        raise DriveAuthError("Google did not return a refresh token")
    save_refresh_token(token_file, refresh)


# -- indexing ------------------------------------------------------------------------------


class DriveIndexer:
    def __init__(self, auth: DriveAuth, root: str, http: httpx2.AsyncClient | None = None) -> None:
        self.auth = auth
        self.root = root.strip("/")
        self._http = http or httpx2.AsyncClient(timeout=60)

    async def _list(self, q: str) -> list[dict[str, Any]]:
        token = await self.auth.access_token()
        files: list[dict[str, Any]] = []
        page: str | None = None
        while True:
            params = {
                "q": q,
                "fields": f"nextPageToken, files({FILE_FIELDS})",
                "pageSize": "1000",
                "spaces": "drive",
            }
            if page:
                params["pageToken"] = page
            resp = await self._http.get(
                f"{DRIVE_API}/files",
                params=params,
                headers={"Authorization": f"Bearer {token.token}"},
            )
            if resp.status_code >= 400:
                raise DriveError(f"Drive API error {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            files.extend(body.get("files", []))
            page = body.get("nextPageToken")
            if not page:
                return files

    async def _children(self, folder_id: str) -> list[dict[str, Any]]:
        return await self._list(f"'{folder_id}' in parents and trashed = false")

    async def _root_id(self) -> str:
        folder_id = "root"
        for name in self.root.split("/"):
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            found = await self._list(
                f"'{folder_id}' in parents and name = '{escaped}' and mimeType = '{FOLDER}'"
                " and trashed = false"
            )
            if not found:
                raise DriveError(f"Folder My Drive/{self.root} not found")
            folder_id = found[0]["id"]
        return folder_id

    async def scan(self) -> list[dict[str, Any]]:
        """Walk the Drive root and return asset rows for the `assets` table."""
        assets: list[dict[str, Any]] = []
        root_id = await self._root_id()
        for top in await self._children(root_id):
            kind = KINDS.get(top["name"])
            if kind is None or top["mimeType"] != FOLDER:
                continue
            await self._walk(top["id"], [top["name"]], kind, assets)
        return assets

    async def _walk(
        self, folder_id: str, parts: list[str], kind: str, out: list[dict[str, Any]]
    ) -> None:
        children = await self._children(folder_id)
        # A diffusers-format directory is a single asset.
        if len(parts) >= 3 and any(c["name"] == "model_index.json" for c in children):
            size = sum(int(c.get("size", 0)) for c in children if c["mimeType"] != FOLDER)
            out.append(self._asset(parts, kind, folder_id, size, None, None))
            return
        for child in children:
            path = [*parts, child["name"]]
            if child["mimeType"] == FOLDER:
                await self._walk(child["id"], path, kind, out)
            elif child["name"].lower().endswith(WEIGHT_SUFFIXES):
                out.append(
                    self._asset(
                        path,
                        kind,
                        child["id"],
                        int(child["size"]) if "size" in child else None,
                        child.get("modifiedTime"),
                        child.get("md5Checksum"),
                    )
                )

    @staticmethod
    def _asset(
        parts: list[str],
        kind: str,
        file_id: str,
        size: int | None,
        mtime: str | None,
        md5: str | None,
    ) -> dict[str, Any]:
        family = parts[1] if kind != "preprocessor" and len(parts) > 2 else None
        return {
            "path": "/".join(parts),
            "family": family,
            "kind": kind,
            "drive_file_id": file_id,
            "size": size,
            "mtime": mtime,
            "md5": md5,
            "sidecar": None,
        }

    async def aclose(self) -> None:
        await self._http.aclose()
