"""The metadata store's tables, and how an older database is brought up to date."""

import sqlite3

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
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint TEXT PRIMARY KEY,
    keys TEXT NOT NULL,
    created_at TEXT NOT NULL
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
    ("assets", "sha256", "TEXT"),
    ("jobs", "runtime", "TEXT"),
    ("results", "segments", "TEXT"),
    ("library_items", "duration", "REAL"),
)

# Stored SDXL params from before the noise schedule was its own control: a Karras sampler
# becomes its plain one plus schedule "karras", and every other sampler gets "default". Mirrors
# `degas.families.sdxl.upgrade_params`; `{col}` is jobs.spec or library_items.config.
SDXL_SCHEDULE_UPGRADE = """
UPDATE {table} SET {col} = json_set(
    {col},
    '$.params.schedule',
    CASE json_extract({col}, '$.params.scheduler')
        WHEN 'dpmpp_2m_karras' THEN 'karras' ELSE 'default' END,
    '$.params.scheduler',
    CASE json_extract({col}, '$.params.scheduler')
        WHEN 'dpmpp_2m_karras' THEN 'dpmpp_2m' ELSE json_extract({col}, '$.params.scheduler') END
)
WHERE json_extract({col}, '$.family') = 'sdxl'
    AND json_extract({col}, '$.params.scheduler') IS NOT NULL
    AND json_extract({col}, '$.params.schedule') IS NULL
"""


def migrate(conn: sqlite3.Connection) -> None:
    """Create missing tables, add columns newer than their table, and upgrade stored specs."""
    conn.executescript(SCHEMA)
    for table, column, type_ in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_}")
    for table, col in (("jobs", "spec"), ("library_items", "config")):
        conn.execute(SDXL_SCHEDULE_UPGRADE.format(table=table, col=col))
