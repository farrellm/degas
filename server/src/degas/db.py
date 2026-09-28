"""SQLite metadata store.

Single-user and low volume: one connection, used from the event loop thread.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    gpu TEXT NOT NULL,
    high_mem INTEGER NOT NULL,
    state TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    last_activity_at TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT NOT NULL,
    queue_position INTEGER NOT NULL,
    spec TEXT NOT NULL,
    seeds TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    log TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, queue_position);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    blob_sha TEXT NOT NULL,
    media_type TEXT NOT NULL,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    duration REAL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    UNIQUE (job_id, item_index)
);
CREATE TABLE IF NOT EXISTS blob_refs (
    blob_sha TEXT NOT NULL,
    ref_type TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    expires_at TEXT,
    PRIMARY KEY (blob_sha, ref_type, ref_id)
);
CREATE TABLE IF NOT EXISTS assets (
    path TEXT PRIMARY KEY,
    family TEXT,
    kind TEXT NOT NULL,
    drive_file_id TEXT NOT NULL,
    size INTEGER,
    mtime TEXT,
    md5 TEXT,
    sidecar TEXT,
    sidecar_rev TEXT,
    preview_thumb TEXT,
    preview_rev TEXT,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS library_items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    media_type TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    config TEXT NOT NULL,
    title TEXT,
    tags TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_result_id TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS library_created ON library_items (created_at);
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    family TEXT,
    tags TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blob_transforms (
    derived_sha TEXT PRIMARY KEY,
    original_sha TEXT NOT NULL,
    ops TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Columns added after a table was first created: (table, column, type).
MIGRATIONS = (
    ("assets", "sidecar_rev", "TEXT"),
    ("assets", "preview_rev", "TEXT"),
    ("jobs", "runtime", "TEXT"),
    ("results", "segments", "TEXT"),
    ("library_items", "duration", "REAL"),
)

ACTIVE_SESSION_STATES = ("starting", "ready", "busy", "stopping")
RESULT_TTL = timedelta(hours=24)
# Uploads, URL imports, frames and transformed images that no job or kept item holds yet.
INPUT_TTL = timedelta(hours=24)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _row(row: sqlite3.Row | None, json_cols: Sequence[str] = ()) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for col in json_cols:
        if d.get(col) is not None:
            d[col] = json.loads(d[col])
    return d


class Database:
    def __init__(self, path: Path | str) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        for table, column, type_ in MIGRATIONS:
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_}")

    def close(self) -> None:
        self.conn.close()

    def _update(self, table: str, id_: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE {table} SET {cols} WHERE id = ?", [*fields.values(), id_])  # noqa: S608

    # -- sessions --------------------------------------------------------------------------

    def create_session(self, gpu: str, high_mem: bool) -> dict[str, Any]:
        id_, ts = new_id(), now()
        self.conn.execute(
            "INSERT INTO sessions (id, gpu, high_mem, state, started_at, last_activity_at)"
            " VALUES (?, ?, ?, 'starting', ?, ?)",
            (id_, gpu, int(high_mem), ts, ts),
        )
        session = self.get_session(id_)
        assert session is not None
        return session

    def get_session(self, id_: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (id_,)).fetchone()
        return _session(row)

    def latest_session(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return _session(row)

    def update_session(self, id_: str, **fields: Any) -> None:
        self._update("sessions", id_, fields)

    def end_session(self, id_: str, state: str, error: str | None = None) -> None:
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

    def insert_job(self, spec: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
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

    def get_job(self, id_: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (id_,)).fetchone()
        return _row(row, JOB_JSON)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
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
        return [_job(r) for r in [*active, *done]]

    def jobs_with_status(self, status: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY queue_position", (status,)
        ).fetchall()
        return [_job(r) for r in rows]

    def next_queued(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY queue_position LIMIT 1"
        ).fetchone()
        return _row(row, JOB_JSON)

    def count_pending(self) -> int:
        (n,) = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(n)

    def update_job(self, id_: str, **fields: Any) -> None:
        if "runtime" in fields:
            fields["runtime"] = json.dumps(fields["runtime"])
        self._update("jobs", id_, fields)

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
        segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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

    def get_result(self, id_: str) -> dict[str, Any] | None:
        row = self.conn.execute(RESULTS_QUERY + " AND r.id = ?", (id_,)).fetchone()
        return _row(row, RESULT_JSON)

    def result_for_blob(self, sha: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            RESULTS_QUERY + " AND r.blob_sha = ? ORDER BY r.created_at DESC LIMIT 1", (sha,)
        ).fetchone()
        return _row(row, RESULT_JSON)

    def has_result(self, job_id: str, item_index: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM results WHERE job_id = ? AND item_index = ?", (job_id, item_index)
        ).fetchone()
        return row is not None

    def list_results(
        self, before: str | None = None, limit: int = 60, job_id: str | None = None
    ) -> list[dict[str, Any]]:
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
        return [r for row in rows if (r := _row(row, RESULT_JSON)) is not None]

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

    def get_transform(self, derived: str) -> dict[str, Any] | None:
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
        with self.conn:
            self.conn.execute("BEGIN")
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

    # -- library ---------------------------------------------------------------------------

    def insert_library_item(
        self, result: dict[str, Any], config: dict[str, Any], inputs: Iterable[str] = ()
    ) -> dict[str, Any]:
        """Keep a result: the item holds its own refs to the image and every input."""
        id_ = new_id()
        kind = "video" if result["media_type"].startswith("video/") else "image"
        with self.conn:
            self.conn.execute("BEGIN")
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

    def get_library_item(self, id_: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM library_items WHERE id = ?", (id_,)).fetchone()
        return _row(row, LIBRARY_JSON)

    def library_item_for_blob(self, sha: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM library_items WHERE blob_sha = ? ORDER BY created_at DESC LIMIT 1",
            (sha,),
        ).fetchone()
        return _row(row, LIBRARY_JSON)

    def library_item_for_result(self, result_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM library_items WHERE source_result_id = ?", (result_id,)
        ).fetchone()
        return _row(row, LIBRARY_JSON)

    def list_library(
        self, query: str | None = None, before: str | None = None, limit: int = 60
    ) -> list[dict[str, Any]]:
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
        return [i for r in rows if (i := _row(r, LIBRARY_JSON)) is not None]

    def update_library_item(self, id_: str, **fields: Any) -> None:
        if "tags" in fields:
            fields["tags"] = json.dumps(fields["tags"], ensure_ascii=False)
        self._update("library_items", id_, fields)

    def delete_library_item(self, id_: str) -> list[str]:
        """Delete a kept item; returns the blobs it held."""
        with self.conn:
            self.conn.execute("BEGIN")
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
    ) -> dict[str, Any]:
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

    def get_prompt(self, id_: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM prompts WHERE id = ?", (id_,)).fetchone()
        return _row(row, ("tags",))

    def list_prompts(self, query: str | None = None) -> list[dict[str, Any]]:
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
        return [p for r in rows if (p := _row(r, ("tags",))) is not None]

    def update_prompt(self, id_: str, **fields: Any) -> None:
        if "tags" in fields:
            fields["tags"] = json.dumps(fields["tags"], ensure_ascii=False)
        self._update("prompts", id_, fields)

    def delete_prompt(self, id_: str) -> bool:
        return self.conn.execute("DELETE FROM prompts WHERE id = ?", (id_,)).rowcount > 0

    # -- assets ----------------------------------------------------------------------------

    def replace_assets(self, assets: Iterable[dict[str, Any]]) -> int:
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
                ts,
            )
            for a in assets
        ]
        with self.conn:
            self.conn.execute("BEGIN")
            self.conn.execute("DELETE FROM assets")
            self.conn.executemany(
                "INSERT INTO assets (path, family, kind, drive_file_id, size, mtime, md5,"
                " sidecar, sidecar_rev, preview_thumb, preview_rev, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def list_assets(
        self, family: str | None = None, kind: str | None = None
    ) -> list[dict[str, Any]]:
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
        return [a for r in rows if (a := _row(r, ("sidecar",))) is not None]

    def get_asset(self, path: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM assets WHERE path = ?", (path,)).fetchone()
        return _row(row, ("sidecar",))

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


JOB_JSON = ("spec", "seeds", "runtime")
LIBRARY_JSON = ("config", "tags")
RESULT_JSON = ("segments",)

# Results with the library item that keeps each one, if any.
RESULTS_QUERY = (
    "SELECT r.*, l.id AS library_id FROM results r"
    " LEFT JOIN library_items l ON l.source_result_id = r.id WHERE 1 = 1"
)


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _session(row: sqlite3.Row | None) -> dict[str, Any] | None:
    d = _row(row)
    if d is not None:
        d["high_mem"] = bool(d["high_mem"])
    return d


def _job(row: sqlite3.Row) -> dict[str, Any]:
    d = _row(row, JOB_JSON)
    assert d is not None
    return d
