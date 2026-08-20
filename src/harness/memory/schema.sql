-- Working and persistent memory.
--
-- This file describes version 1 only. It is applied to a fresh database and
-- never edited to describe a later version: an existing database is moved
-- forward by the migration table in store.py, so a schema written by an older
-- harness is detected and upgraded rather than reinterpreted in place.

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    goal              TEXT    NOT NULL,
    workspace         TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    stop_reason       TEXT,
    answer            TEXT,
    verified          INTEGER NOT NULL DEFAULT 0,
    steps_taken       INTEGER NOT NULL DEFAULT 0,
    tool_calls        INTEGER NOT NULL DEFAULT 0,
    retries           INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,
    elapsed_s         REAL    NOT NULL DEFAULT 0.0
);

-- One row per attempted step, written before the step runs. Retries of the
-- same step_index add rows rather than overwriting, so the history of a run
-- stays honest about how often it had to try.
CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step_index  INTEGER NOT NULL,
    role        TEXT,
    action      TEXT    NOT NULL,
    tool        TEXT,
    arguments   TEXT,
    status      TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    finished_at TEXT,
    duration_ms REAL,
    exit_code   INTEGER,
    error_kind  TEXT,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS steps_run_idx ON steps(run_id, step_index, id);

-- The resume point: the whole TaskState as JSON, one row per run.
CREATE TABLE IF NOT EXISTS task_state (
    run_id     TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    state_json TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT    NOT NULL UNIQUE,
    value      TEXT    NOT NULL,
    category   TEXT    NOT NULL DEFAULT 'general',
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS facts_category_idx ON facts(category, key);
