-- Schema version 1, kept verbatim so the upgrade path has a real predecessor
-- to migrate rather than a reconstruction of one. Version 2 adds
-- runtime_state; everything else here is what a v1 harness wrote.

CREATE TABLE runs (
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

CREATE TABLE steps (
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

CREATE INDEX steps_run_idx ON steps(run_id, step_index, id);

CREATE TABLE task_state (
    run_id     TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    state_json TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE TABLE facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT    NOT NULL UNIQUE,
    value      TEXT    NOT NULL,
    category   TEXT    NOT NULL DEFAULT 'general',
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE INDEX facts_category_idx ON facts(category, key);

PRAGMA user_version = 1;
