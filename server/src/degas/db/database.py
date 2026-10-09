"""The metadata store's queries.

Single-user and low volume: one connection, used from the event loop thread.
"""

import contextlib
import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Unpack, cast

from degas.db.rows import (
    AssetRow,
    Cleared,
    JobRow,
    JobStatus,
    JobUpdate,
    LibraryItem,
    LibraryItemUpdate,
    PromptUpdate,
    PushSubscriptionRow,
    ResultRow,
    SavedConfig,
    SavedPrompt,
    SessionRow,
    SessionState,
    SessionUpdate,
    TransformRecord,
)
from degas.db.schema import migrate
from degas_worker.spec import Spec

ACTIVE_SESSION_STATES = ("starting", "ready", "busy", "stopping")
RUNNING_SESSION_STATES = ("ready", "busy")  # the worker is up
PENDING_JOB_STATUSES = ("queued", "running")
RESULT_TTL = timedelta(hours=24)
# Uploads, URL imports, frames and transformed images that no job or kept item holds yet.
INPUT_TTL = timedelta(hours=24)

# Columns stored as JSON text.
JOB_JSON = ("spec", "seeds", "runtime")
LIBRARY_JSON = ("config", "tags")
RESULT_JSON = ("segments",)
ASSET_JSON = ("sidecar",)
PROMPT_JSON = ("tags",)

# Results with the library item that keeps each one, if any.
RESULTS_QUERY = (
    "SELECT r.*, l.id AS library_id FROM results r"
    " LEFT JOIN library_items l ON l.source_result_id = r.id WHERE 1 = 1"
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _decode(row: sqlite3.Row, json_cols: Sequence[str] = ()) -> dict[str, Any]:
    d = dict(row)
    for col in json_cols:
        if d.get(col) is not None:
            d[col] = json.loads(d[col])
    return d


def _row[T](row: sqlite3.Row | None, kind: type[T], json_cols: Sequence[str] = ()) -> T | None:
    """A fetched row as its TypedDict (`kind` is only for the type checker)."""
    return None if row is None else cast("T", _decode(row, json_cols))


def _rows[T](rows: Iterable[sqlite3.Row], kind: type[T], json_cols: Sequence[str] = ()) -> list[T]:
    return [cast("T", _decode(row, json_cols)) for row in rows]


def _session(row: sqlite3.Row | None) -> SessionRow | None:
    session = _row(row, SessionRow)
    if session is not None:
        session["high_mem"] = bool(session["high_mem"])
    return session


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Database:
    def __init__(self, path: Path | str) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Several statements as one atomic change (the connection autocommits otherwise)."""
        with self.conn:
            self.conn.execute("BEGIN")
            yield

    def _update(self, table: str, id_: str, fields: Mapping[str, Any]) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE {table} SET {cols} WHERE id = ?", [*fields.values(), id_])  # noqa: S608

    # -- sessions --------------------------------------------------------------------------

    def create_session(self, gpu: str, high_mem: bool) -> SessionRow:
        id_, ts = new_id(), now()
        self.conn.execute(
            "INSERT INTO sessions (id, gpu, high_mem, state, started_at, last_activity_at)"
            " VALUES (?, ?, ?, 'starting', ?, ?)",
            (id_, gpu, int(high_mem), ts, ts),
        )
        session = self.get_session(id_)
        assert session is not None
        return session

    def get_session(self, id_: str) -> SessionRow | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (id_,)).fetchone()
        return _session(row)

    def latest_session(self) -> SessionRow | None:
        row = self.conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return _session(row)

    def update_session(self, id_: str, **fields: Unpack[SessionUpdate]) -> None:
        self._update("sessions", id_, fields)

    def end_session(self, id_: str, state: SessionState, error: str | None = None) -> None:
        """Mark a session ended; its results start their retention clock."""
        ended = now()
        self.update_session(id_, state=state, ended_at=ended, error=error)
        expires = (datetime.fromisoformat(ended) + RESULT_TTL).isoformat(timespec="milliseconds")
        self.conn.execute(
            "UPDATE results SET expires_at = ? WHERE expires_at IS NULL AND job_id IN"
            " (SELECT id FROM jobs WHERE session_id = ?)",
            (expires, id_),
        )

    # -- jobs ------------------------------------------------------------------------------

    def insert_job(self, spec: Spec, seeds: list[int]) -> JobRow:
        id_ = new_id()
        (pos,) = self.conn.execute(
            "SELECT COALESCE(MAX(queue_position), 0) + 1 FROM jobs"
        ).fetchone()
        self.conn.execute(
            "INSERT INTO jobs (id, status, queue_position, spec, seeds, created_at)"
            " VALUES (?, 'queued', ?, ?, ?, ?)",
            (id_, pos, json.dumps(spec), json.dumps(seeds), now()),
        )
        job = self.get_job(id_)
        assert job is not None
        return job

    def get_job(self, id_: str) -> JobRow | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (id_,)).fetchone()
        return _row(row, JobRow, JOB_JSON)

    def list_jobs(self, limit: int = 50) -> list[JobRow]:
        """Queued and running jobs (in queue order), then the most recent finished ones."""
        active = self.conn.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'running')"
            " ORDER BY status = 'queued', queue_position"
        ).fetchall()
        done = self.conn.execute(
            "SELECT * FROM jobs WHERE status NOT IN ('queued', 'running')"
            " ORDER BY finished_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return _rows([*active, *done], JobRow, JOB_JSON)

    def jobs_with_status(self, status: JobStatus) -> list[JobRow]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY queue_position", (status,)
        ).fetchall()
        return _rows(rows, JobRow, JOB_JSON)

    def next_queued(self) -> JobRow | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_position LIMIT 1"
        ).fetchone()
        return _row(row, JobRow, JOB_JSON)

    def move_job(self, id_: str, index: int) -> list[str]:
        """Move a queued job to `index` in the queue (0 is next). Returns the queue's ids.

        The queued jobs swap their existing positions, so later submissions still go last.
        """
        rows = self.conn.execute(
            "SELECT id, queue_position FROM jobs WHERE status = 'queued' ORDER BY queue_position"
        ).fetchall()
        ids = [r["id"] for r in rows]
        if id_ not in ids:
            return ids
        ids.remove(id_)
        ids.insert(max(0, min(index, len(ids))), id_)
        with self._transaction():
            self.conn.executemany(
                "UPDATE jobs SET queue_position = ? WHERE id = ?",
                [(r["queue_position"], job) for r, job in zip(rows, ids, strict=True)],
            )
        return ids

    def count_pending(self) -> int:
        (n,) = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(n)

    def update_job(self, id_: str, **fields: Unpack[JobUpdate]) -> None:
        encoded: dict[str, Any] = {**fields}
        if "runtime" in fields:
            encoded["runtime"] = json.dumps(fields["runtime"])
        self._update("jobs", id_, encoded)

    # -- results ---------------------------------------------------------------------------

    def insert_result(
        self,
        job_id: str,
        item_index: int,
        blob_sha: str,
        media_type: str,
        seed: int | None,
        width: int | None,
        height: int | None,
        duration: float | None = None,
        segments: list[SavedConfig] | None = None,
    ) -> ResultRow:
        """`segments`: for a stitched video, the configs of the clips it chains (design §6.4)."""
        id_ = new_id()
        self.conn.execute(
            "INSERT INTO results (id, job_id, item_index, blob_sha, media_type, seed, width,"
            " height, duration, segments, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                id_,
                job_id,
                item_index,
                blob_sha,
                media_type,
                seed,
                width,
                height,
                duration,
                json.dumps(segments) if segments is not None else None,
                now(),
            ),
        )
        self.add_blob_ref(blob_sha, "result", id_)
        result = self.get_result(id_)
        assert result is not None
        return result

    def get_result(self, id_: str) -> ResultRow | None:
        row = self.conn.execute(RESULTS_QUERY + " AND r.id = ?", (id_,)).fetchone()
        return _row(row, ResultRow, RESULT_JSON)

    def result_for_blob(self, sha: str) -> ResultRow | None:
        row = self.conn.execute(
            RESULTS_QUERY + " AND r.blob_sha = ? ORDER BY r.created_at DESC LIMIT 1", (sha,)
        ).fetchone()
        return _row(row, ResultRow, RESULT_JSON)

    def has_result(self, job_id: str, item_index: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM results WHERE job_id = ? AND item_index = ?", (job_id, item_index)
        ).fetchone()
        return row is not None

    def list_results(
        self, before: str | None = None, limit: int = 60, job_id: str | None = None
    ) -> list[ResultRow]:
        """Newest first. `before` is a `created_at` cursor."""
        query = RESULTS_QUERY
        args: list[Any] = []
        if before:
            query += " AND r.created_at < ?"
            args.append(before)
        if job_id:
            query += " AND r.job_id = ?"
            args.append(job_id)
        query += " ORDER BY r.created_at DESC, r.item_index DESC LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(query, args).fetchall()
        return _rows(rows, ResultRow, RESULT_JSON)

    def add_blob_ref(
        self, sha: str, ref_type: str, ref_id: str, expires_at: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO blob_refs (blob_sha, ref_type, ref_id, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (sha, ref_type, ref_id, expires_at),
        )

    def hold_input(self, sha: str, ttl: timedelta = INPUT_TTL) -> None:
        """Keep an input blob for `ttl` from now (jobs and kept items take their own refs)."""
        expires = (datetime.now(UTC) + ttl).isoformat(timespec="milliseconds")
        self.conn.execute(
            "INSERT INTO blob_refs (blob_sha, ref_type, ref_id, expires_at)"
            " VALUES (?, 'input', ?, ?) ON CONFLICT (blob_sha, ref_type, ref_id)"
            " DO UPDATE SET expires_at = MAX(expires_at, excluded.expires_at)",
            (sha, sha, expires),
        )

    # -- transforms (design §6.5) ----------------------------------------------------------

    def add_transform(self, derived: str, original: str, ops: list[dict[str, Any]]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO blob_transforms (derived_sha, original_sha, ops)"
            " VALUES (?, ?, ?)",
            (derived, original, json.dumps(ops)),
        )

    def get_transform(self, derived: str) -> TransformRecord | None:
        row = self.conn.execute(
            "SELECT original_sha, ops FROM blob_transforms WHERE derived_sha = ?", (derived,)
        ).fetchone()
        return None if row is None else {"original": row[0], "ops": json.loads(row[1])}

    def forget_transforms(self, shas: Iterable[str]) -> None:
        self.conn.executemany(
            "DELETE FROM blob_transforms WHERE derived_sha = ?", [(s,) for s in shas]
        )

    def remove_blob_refs(self, ref_type: str, ref_id: str) -> list[str]:
        """Drop one referrer's refs; returns the blobs it held."""
        rows = self.conn.execute(
            "DELETE FROM blob_refs WHERE ref_type = ? AND ref_id = ? RETURNING blob_sha",
            (ref_type, ref_id),
        ).fetchall()
        return [r[0] for r in rows]

    def is_referenced(self, sha: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM blob_refs WHERE blob_sha = ?", (sha,)).fetchone()
        return row is not None

    def referenced_blobs(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT DISTINCT blob_sha FROM blob_refs")}

    # -- retention (design §6.3) -----------------------------------------------------------

    def expire(self, at: str | None = None) -> dict[str, int]:
        """Delete expired results, and finished jobs left with nothing to show.

        Blobs are not touched here: whatever lost its last reference is removed by
        the blob sweep afterwards.
        """
        at = at or now()
        cutoff = (datetime.fromisoformat(at) - RESULT_TTL).isoformat(timespec="milliseconds")
        with self._transaction():
            results = self.conn.execute(
                "DELETE FROM results WHERE expires_at IS NOT NULL AND expires_at <= ? RETURNING id",
                (at,),
            ).fetchall()
            self.conn.executemany(
                "DELETE FROM blob_refs WHERE ref_type = 'result' AND ref_id = ?",
                [(r[0],) for r in results],
            )
            jobs = self.conn.execute(
                "DELETE FROM jobs WHERE status NOT IN ('queued', 'running')"
                " AND finished_at <= ? AND id NOT IN (SELECT job_id FROM results)"
                " RETURNING id",
                (cutoff,),
            ).fetchall()
            self.conn.executemany(
                "DELETE FROM blob_refs WHERE ref_type = 'job' AND ref_id = ?",
                [(r[0],) for r in jobs],
            )
            refs = self.conn.execute(
                "DELETE FROM blob_refs WHERE expires_at IS NOT NULL AND expires_at <= ?", (at,)
            ).rowcount
        return {"results": len(results), "jobs": len(jobs), "refs": refs}

    def clear_results(self) -> Cleared:
        """Delete every finished job and its results now, without waiting for them to expire.

        Queued and running jobs stay. Kept images hold their own refs in the library.
        Returns the counts and the blobs that lost a reference.
        """
        with self._transaction():
            jobs = [
                r[0]
                for r in self.conn.execute(
                    "DELETE FROM jobs WHERE status NOT IN ('queued', 'running') RETURNING id"
                ).fetchall()
            ]
            results = [
                r[0]
                for r in self.conn.execute(
                    "DELETE FROM results WHERE job_id NOT IN (SELECT id FROM jobs) RETURNING id"
                ).fetchall()
            ]
            return self._release(jobs, results)

    def delete_job_results(self, job_id: str, chain: bool) -> Cleared:
        """Delete one finished job's results now: its stitched chain, or everything else.

        The job goes too once nothing is left to show. Returns the counts and the blobs
        that lost a reference.
        """
        with self._transaction():
            results = [
                r[0]
                for r in self.conn.execute(
                    "DELETE FROM results WHERE job_id = ? AND (segments IS NOT NULL) = ?"
                    " RETURNING id",
                    (job_id, chain),
                ).fetchall()
            ]
            jobs = [
                r[0]
                for r in self.conn.execute(
                    "DELETE FROM jobs WHERE id = ? AND status NOT IN ('queued', 'running')"
                    " AND id NOT IN (SELECT job_id FROM results) RETURNING id",
                    (job_id,),
                ).fetchall()
            ]
            return self._release(jobs, results)

    def _release(self, jobs: list[str], results: list[str]) -> Cleared:
        """Drop deleted jobs' and results' blob refs; returns the counts and those blobs."""
        shas: set[str] = set()
        for ref_type, ids in (("job", jobs), ("result", results)):
            for id_ in ids:
                shas.update(self.remove_blob_refs(ref_type, id_))
        return {"results": len(results), "jobs": len(jobs), "blobs": sorted(shas)}

    # -- library ---------------------------------------------------------------------------

    def insert_library_item(
        self, result: ResultRow, config: SavedConfig, inputs: Iterable[str] = ()
    ) -> LibraryItem:
        """Keep a result: the item holds its own refs to the image and every input."""
        id_ = new_id()
        kind = "video" if result["media_type"].startswith("video/") else "image"
        with self._transaction():
            self.conn.execute(
                "INSERT INTO library_items (id, kind, blob_sha, media_type, width, height,"
                " duration, config, title, tags, created_at, source_result_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '[]', ?, ?)",
                (
                    id_,
                    kind,
                    result["blob_sha"],
                    result["media_type"],
                    result["width"],
                    result["height"],
                    result.get("duration"),
                    json.dumps(config),
                    now(),
                    result["id"],
                ),
            )
            for sha in {result["blob_sha"], *inputs}:
                self.add_blob_ref(sha, "library", id_)
        item = self.get_library_item(id_)
        assert item is not None
        return item

    def get_library_item(self, id_: str) -> LibraryItem | None:
        row = self.conn.execute("SELECT * FROM library_items WHERE id = ?", (id_,)).fetchone()
        return _row(row, LibraryItem, LIBRARY_JSON)

    def library_item_for_blob(self, sha: str) -> LibraryItem | None:
        row = self.conn.execute(
            "SELECT * FROM library_items WHERE blob_sha = ? ORDER BY created_at DESC LIMIT 1",
            (sha,),
        ).fetchone()
        return _row(row, LibraryItem, LIBRARY_JSON)

    def library_item_for_result(self, result_id: str) -> LibraryItem | None:
        row = self.conn.execute(
            "SELECT * FROM library_items WHERE source_result_id = ?", (result_id,)
        ).fetchone()
        return _row(row, LibraryItem, LIBRARY_JSON)

    def list_library(
        self, query: str | None = None, before: str | None = None, limit: int = 60
    ) -> list[LibraryItem]:
        """Newest first, optionally matching prompt text, title or tags."""
        sql = "SELECT * FROM library_items WHERE 1 = 1"
        args: list[Any] = []
        for word in (query or "").split():
            like = f"%{_escape_like(word)}%"
            sql += (
                " AND (json_extract(config, '$.params.prompt') LIKE ? ESCAPE '\\'"
                " OR json_extract(config, '$.params.negative_prompt') LIKE ? ESCAPE '\\'"
                " OR title LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')"
            )
            args += [like] * 4
        if before:
            sql += " AND created_at < ?"
            args.append(before)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = self.conn.execute(sql, args).fetchall()
        return _rows(rows, LibraryItem, LIBRARY_JSON)

    def update_library_item(self, id_: str, **fields: Unpack[LibraryItemUpdate]) -> None:
        encoded: dict[str, Any] = {**fields}
        if "tags" in fields:
            encoded["tags"] = json.dumps(fields["tags"], ensure_ascii=False)
        self._update("library_items", id_, encoded)

    def delete_library_item(self, id_: str) -> list[str]:
        """Delete a kept item; returns the blobs it held."""
        with self._transaction():
            self.conn.execute("DELETE FROM library_items WHERE id = ?", (id_,))
            return self.remove_blob_refs("library", id_)

    # -- saved prompts ---------------------------------------------------------------------

    def insert_prompt(
        self,
        name: str,
        prompt: str,
        negative_prompt: str,
        family: str | None,
        tags: list[str],
    ) -> SavedPrompt:
        id_ = new_id()
        self.conn.execute(
            "INSERT INTO prompts (id, name, prompt, negative_prompt, family, tags, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                id_,
                name,
                prompt,
                negative_prompt,
                family,
                json.dumps(tags, ensure_ascii=False),
                now(),
            ),
        )
        saved = self.get_prompt(id_)
        assert saved is not None
        return saved

    def get_prompt(self, id_: str) -> SavedPrompt | None:
        row = self.conn.execute("SELECT * FROM prompts WHERE id = ?", (id_,)).fetchone()
        return _row(row, SavedPrompt, PROMPT_JSON)

    def list_prompts(self, query: str | None = None) -> list[SavedPrompt]:
        sql = "SELECT * FROM prompts WHERE 1 = 1"
        args: list[Any] = []
        for word in (query or "").split():
            like = f"%{_escape_like(word)}%"
            sql += (
                " AND (name LIKE ? ESCAPE '\\' OR prompt LIKE ? ESCAPE '\\'"
                " OR negative_prompt LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')"
            )
            args += [like] * 4
        sql += " ORDER BY created_at DESC"
        rows = self.conn.execute(sql, args).fetchall()
        return _rows(rows, SavedPrompt, PROMPT_JSON)

    def update_prompt(self, id_: str, **fields: Unpack[PromptUpdate]) -> None:
        encoded: dict[str, Any] = {**fields}
        if "tags" in fields:
            encoded["tags"] = json.dumps(fields["tags"], ensure_ascii=False)
        self._update("prompts", id_, encoded)

    def delete_prompt(self, id_: str) -> bool:
        return self.conn.execute("DELETE FROM prompts WHERE id = ?", (id_,)).rowcount > 0

    # -- assets ----------------------------------------------------------------------------

    def replace_assets(self, assets: Iterable[Mapping[str, Any]]) -> int:
        """Replace the Drive index. Preview images are held by `asset` blob refs."""
        ts = now()
        rows = [
            (
                a["path"],
                a.get("family"),
                a["kind"],
                a["drive_file_id"],
                a.get("size"),
                a.get("mtime"),
                a.get("md5"),
                json.dumps(a["sidecar"]) if a.get("sidecar") is not None else None,
                a.get("sidecar_rev"),
                a.get("preview_thumb"),
                a.get("preview_rev"),
                a.get("sha256"),
                ts,
            )
            for a in assets
        ]
        with self._transaction():
            self.conn.execute("DELETE FROM assets")
            self.conn.executemany(
                "INSERT INTO assets (path, family, kind, drive_file_id, size, mtime, md5,"
                " sidecar, sidecar_rev, preview_thumb, preview_rev, sha256, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.execute("DELETE FROM blob_refs WHERE ref_type = 'asset'")
            self.conn.executemany(
                "INSERT OR IGNORE INTO blob_refs (blob_sha, ref_type, ref_id)"
                " VALUES (?, 'asset', ?)",
                [(r[9], r[0]) for r in rows if r[9]],
            )
        self.set_setting("drive.indexed_at", ts)
        return len(rows)

    def list_assets(self, family: str | None = None, kind: str | None = None) -> list[AssetRow]:
        query = "SELECT * FROM assets WHERE 1 = 1"
        args: list[Any] = []
        if family:
            query += " AND family = ?"
            args.append(family)
        if kind:
            query += " AND kind = ?"
            args.append(kind)
        query += " ORDER BY path"
        rows = self.conn.execute(query, args).fetchall()
        return _rows(rows, AssetRow, ASSET_JSON)

    def delete_assets(self, paths: Iterable[str]) -> list[str]:
        """Drop assets from the index; returns the preview blobs they held."""
        shas: list[str] = []
        with self._transaction():
            for path in paths:
                self.conn.execute("DELETE FROM assets WHERE path = ?", (path,))
                shas += self.remove_blob_refs("asset", path)
        return shas

    def get_asset(self, path: str) -> AssetRow | None:
        row = self.conn.execute("SELECT * FROM assets WHERE path = ?", (path,)).fetchone()
        return _row(row, AssetRow, ASSET_JSON)

    # -- push subscriptions ----------------------------------------------------------------

    def add_push_subscription(self, endpoint: str, keys: dict[str, str]) -> None:
        self.conn.execute(
            "INSERT INTO push_subscriptions (endpoint, keys, created_at) VALUES (?, ?, ?)"
            " ON CONFLICT (endpoint) DO UPDATE SET keys = excluded.keys",
            (endpoint, json.dumps(keys), now()),
        )

    def delete_push_subscription(self, endpoint: str) -> bool:
        cur = self.conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
        return cur.rowcount > 0

    def list_push_subscriptions(self) -> list[PushSubscriptionRow]:
        rows = self.conn.execute("SELECT * FROM push_subscriptions ORDER BY created_at").fetchall()
        return _rows(rows, PushSubscriptionRow, ("keys",))

    # -- settings --------------------------------------------------------------------------

    def get_setting(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
