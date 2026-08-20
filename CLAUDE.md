# Claude Code Notes for harness-qwen

This file supplements [AGENTS.md](AGENTS.md), which holds the general working agreements for this repository. Below are Claude-specific practices.

## Before changing context handling

Anyone working on `PromptAssembler`, `TokenBudget`, `CacheEconomics`, or the prompt-building pipeline must read `docs/CONTEXT.md` first. The cache reprocesses the entire prefix (24.83 s per 4.8k tokens at first load) if a single byte changes; cost-benefit decisions for compression are non-obvious and empirically grounded in measured pp_rate curves from `docs/DISCOVERY.md`.

## Verify, do not assert

Show test output. Show `git diff`. Show file contents via `ls -lah` or `head`. Claims that "the test passes" or "this works" are incomplete without captured evidence. Red tests are blockers, not hints.

## Scope discipline

Make surgical changes:
- Edit only files the task requires.
- No refactorings outside the assigned scope.
- No moving files, renaming modules, or reorganizing imports unless the task explicitly asks.
- No "while I'm here" cleanups in files you did not intend to modify.

## Working with the local model

The harness targets a model running at 16.5 tokens per second. Waiting for a full model response during testing is patience work — a 500-token generation takes ~30 seconds. For fast iteration on agent logic, use `FakeProvider` and scripted test fixtures instead.

## Useful commands

| Command | Purpose |
|---------|---------|
| `uv run pytest -m "not local_llm"` | Unit tests without the 35B model (fast CI loop) |
| `uv run pytest -m local_llm` | Full agent tests with the real model (slow, local only) |
| `uv run harness doctor` | Probe hardware profile and runtime readiness |
| `uv run harness benchmark` | Run capability benchmarks from `benchmarks/model-capabilities.json` |
| `uv run pytest --cov=harness --cov-report=json && uv run python scripts/coverage_gate.py` | What CI gates on: one measured run, then the per-package floors |

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage labels with default vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context documentation layout. See `docs/agents/domain.md`.
