# Context Engine — Detailed Specification

This document specifies the mechanics behind the "prefix contract" and "cache
economics" introduced in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), at the
level of detail needed to implement `src/harness/context/`. Every number below
is copied verbatim from [docs/DISCOVERY.md](docs/DISCOVERY.md). Where a value
has not been measured — the context-size sweet spot in particular — this
document says so instead of inventing one.

## 1. The economics

On a cloud endpoint, throughput is high and the provider absorbs concurrency
behind elastic infrastructure. None of that applies here: one process, one
GPU, one slot, on a 2017 quad-core CPU that is already the bottleneck. A
second concurrent slot does not add capacity, it halves throughput (16.5 tok/s
single slot → 8.55 tok/s with a second slot active), so the harness runs
`parallel_model_requests = 1`. There is nothing to hide a slow step behind.

There was also a real risk that caching would not work at all: llama.cpp issue
#20225 reports that hybrid attention/recurrent models must reprocess the whole
prompt every turn, because the recurrent (Gated DeltaNet) state cannot be
partially rewound — for a model where 30 of 40 layers are recurrent, that
would mean ~106 s per step at 16k tokens, always. Measured against this
server, that pathological case does **not** apply:

| Run | Tokens reprocessed | Wall clock |
|---|---|---|
| 1 — cold, 4816 tokens | 4816 | 24.83 s |
| 2 — byte-identical prompt | 4 | 0.49 s |
| 3 — same prefix + 11 new tokens | 11 | 0.82 s |
| 4 — same prefix + 22 new tokens (11 new since run 3) | 11 | 0.89 s |

Incremental caching works, at roughly **50× speedup** between cold reprocess
and cache hit. Even a byte-identical prompt still reprocesses 4 tokens — the
floor is not zero, but appending 11 more tokens still costs only ~0.8 s,
regardless of how large the existing prefix is.

Prompt-processing throughput is not constant, though — it degrades with size:

| Context size | pp rate |
|---|---|
| cold, 4816-token probe | 194 tok/s |
| 2k–6k tokens | ~170 tok/s |
| 16k tokens | 154 tok/s |
| 18k+ tokens | ~120 tok/s |

**What an 8k reprocess costs, explicitly.** ARCHITECTURE.md gives ~50 s as the
reprocess time at 8k context. At the measured single-slot generation rate:

```
50 s × 16.5 tok/s = 825 tokens of generation that could have happened instead
```

That is the source of the "~825 discarded output tokens" figure — 825 tokens
of real model output the run did not get to produce. The same arithmetic at
16k, using DISCOVERY's ~106 s figure: `106 s × 16.5 tok/s ≈ 1749 tokens`.
Rewriting the front of the prompt gets *more* expensive as a run goes on, not
less, because pp-rate falls as context grows while the reprocessed span also
grows. This is why "compress when the context gets large" is backwards as a
default policy — compression itself is the expensive operation. §5 exists
because of this inversion.

## 2. Context zones

Every segment belongs to exactly one of the two zones from ARCHITECTURE.md —
the stable prefix (byte-identical for the run, cached at ~50× after the first
hit) or the append zone (grows only, ~0.8 s per step regardless of size). The
assignment follows directly from whether a segment's content is fixed for the
run or changes per step.

| Segment | Zone | Why |
|---|---|---|
| System / instructions | Stable prefix | Identical across every role and step; no reason to ever pay to reprocess it. |
| Tool schemas | Stable prefix | The registered tool set does not change mid-run under normal operation; declare once, reuse via ~50× cache hit. |
| Repository map | Stable prefix | Expensive to build, cheap to reuse; paid once at the cold-start rate (194 tok/s) and amortised over every later step. |
| Task statement | Stable prefix | Fixed for the run by definition; putting it in the append zone risks an accidental edit invalidating the cache for nothing. |
| Working context (role directive + step history) | Append zone | Changes on every step by construction — forcing it into the prefix means a full reprocess every step. |
| Conversation summary | Append zone | Sits where the older history it replaces used to sit. Producing it is a deliberate, budgeted rewrite (§5); once written it is static append-zone content. |
| Task state | Append zone | Mutates every step (step index, plan progress, open problems); in the prefix it would invalidate the cache almost every step. |
| Retrieved context | Append zone | Produced fresh per step by a retrieval call, inserted at the tail — structurally a tool result. |
| Tool results | Append zone | Inserted after each tool execution per the agent loop; always fresh, always at the tail. |
| Persistent memory | Append zone | Injected selectively via the `Retriever` interface, not loaded wholesale — per-step content, not run-constant content. |

The prefix holds exactly four segments, all genuinely constant; everything
else changes at least once per step and belongs in the append zone.

## 3. Token budget

**The absolute numbers here are not yet measured.** The context-size sweet
spot is answered only by the planned flag sweep (the not-yet-implemented
`src/harness/benchmark/`, per DISCOVERY.md §5.2 and VISION.md), which has not
run against a sized-context matrix. What follows is the *shape* of the budget
plus one placeholder, explicitly marked provisional.

```
C = P + A + G

C  total context window in use (hard ceiling: --ctx-size 65536 per the
   current launch flags, DISCOVERY.md §2 — a configured maximum, not a
   validated working target)
P  stable prefix — whatever system + tools + repo map + task statement
   actually assemble to; measured per run, not chosen as a fraction
A  append zone — everything else; A = C - P - G
G  generation reserve — tokens withheld from input growth so the run
   always has room to produce output, sized against the run's configured
   token_budget (an existing hard bound per ARCHITECTURE.md)
```

`P` cannot be a fraction — it is dictated by the actual size of the tool set
and repository map, measured at assembly time. The two real levers are `G`
and the soft ceiling at which `A`'s growth triggers a compression check.

**Provisional starting point (UNVALIDATED, pending flag sweep):**

- Soft ceiling for total working context: **16,000 tokens** — the last point
  in the measured pp-rate profile before the sharp fall to ~120 tok/s past
  18k. A hint from the data about where processing cost accelerates, not a
  benchmarked optimum.
- Generation reserve: **15% of the soft ceiling** (≈2,400 tokens) — a
  conservative starting fraction, unmeasured.
- Compression trigger: any append that would push `A` past the soft ceiling
  first goes through `CacheEconomics` (§5).

These exist so the loop has something to run before the sweep, not because
they are correct.

## 4. The prefix contract

Byte-stability is an assertion, not a convention. `PromptAssembler` hashes the
assembled stable prefix on every call and compares it to the previous hash. An
undeclared mismatch aborts the run — silent prefix drift is exactly the
failure this contract prevents.

A prefix change is legitimate only when it is:

1. **Declared before assembly**, not discovered as a diff afterward.
2. **Attributable to one of a fixed set of reasons** — task statement changed
   (new goal), a tool was registered or removed mid-run, the repository map
   was explicitly refreshed, or `CacheEconomics` approved a compression pass
   (§5).
3. **Logged with a reason code and its measured cost** — `prefix hash` is
   already a tracked telemetry field per ARCHITECTURE.md.
4. **Charged against the run's budget**, competing with `token_budget` and
   `wall_clock_timeout` like any other spend.

**Role switching stays in the append zone precisely because of this cost.** A
role-specific system prompt would make every handover a prefix change:
ARCHITECTURE.md gives ~50 s at 8k context per handover — ~825 discarded output
tokens (§1), on every single switch. Instead all roles share one system
prompt in the stable prefix, and the role is a directive appended at the
current position — a normal ~0.8 s append, however often it happens.

## 5. Compression cost model

From ARCHITECTURE.md:

```
reprocess_cost  = new_prefix_tokens / pp_rate
saving_per_step = freed_tokens / pp_rate           # only if prefix shrinks
remaining_steps = max_steps - current_step

compress if  saving_per_step * remaining_steps > reprocess_cost * safety_margin
```

`pp_rate` is read from the measured profile in §1, interpolated at the
current size — not a constant.

**Worked example** (`remaining_steps`, `freed_tokens`, `safety_margin` are
illustrative inputs; `pp_rate` is the real measured 18k+ figure):

At step 20 of a `max_steps = 40` run, context has reached 18k tokens. A
candidate compression would summarize old tool output, freeing 4,000 tokens
and requiring the 14,000 tokens after the edit point to be reprocessed.

```
pp_rate (18k+)   = 120 tok/s                         # measured, DISCOVERY.md
reprocess_cost   = 14,000 / 120  ≈ 116.7 s
saving_per_step  =  4,000 / 120  ≈  33.3 s
remaining_steps  = 40 - 20 = 20
safety_margin    = 1.5                                 # placeholder

saving_per_step * remaining_steps = 33.3 * 20 = 666.7 s
reprocess_cost * safety_margin    = 116.7 * 1.5 = 175.0 s

666.7 > 175.0  →  compress
```

At step 38 (`remaining_steps = 2`) the left side drops to 66.7 s against the
same 175 s threshold — the loop refuses, finishes on the current context, and
reports that compression did not amortise, rather than pay for a rewrite with
almost no run left to benefit from it.

**Escalation order, reordered by actual cache cost.** The nominal order (drop
irrelevant tool output → summarize old tool output → summarize old
conversation → retrieve again → only then grow the context window) is cloud
API intuition, where every input token costs about the same regardless of
position. That is false here: cost is driven by how much content sits *after*
the edit point, since everything after an edit must be reprocessed regardless
of which zone it nominally belongs to. Reordered, cheapest first:

1. **Retrieve again** — a pure tail append (~0.8 s, §1), local via FTS5 (§7).
   The nominal order treats this as a late resort; here it never touches
   existing content, so it is the first thing to reach for.
2. **Drop irrelevant tool output** — touches history; cheap if the target is
   recent, expensive if old (and it usually is old — that's what's safe to
   drop).
3. **Summarize old tool output** — same mechanism as #2, typically a larger
   edit region.
4. **Summarize old conversation** — the most expensive in-append-zone edit: by
   definition it targets content closest to the prefix boundary, so the
   largest possible span must be reprocessed.
5. **Grow the context window** — last, but not for the nominal reason. Per
   ARCHITECTURE.md's runtime supervision, this is a server reconfiguration
   requiring graceful shutdown and restart, on top of a full reprocess at a
   *slower* pp_rate than whatever size the run was already at (§1). It is
   strictly worse than every compression option above it, not just less
   preferred.

## 6. Tool result compression

Raw tool output never enters the context. `ToolResult` carries a compressed
view; the full output is written to the run directory and addressable by id
(ARCHITECTURE.md) — nothing is lost, it simply is not paid for on every
subsequent step.

**Kept, always:** exit code, capped stderr, matched/error lines, file paths
touched or referenced, a head-and-tail window of stdout, and an explicit count
of elided lines — never a silent truncation.

**Dropped from context, retained on disk:** the full stdout body, verbose
progress/logging noise, and anything redundant with content already
addressable by id.

| Case | Kept in context | Elided (addressable by id) |
|---|---|---|
| pytest output | exit code, pass/fail summary, full traceback per failing test, file:line references | stdout from passing tests, setup/collection noise |
| `git diff` | changed file paths, per-file +/- stat, full hunks for small files | large hunks beyond a size threshold — head/tail window + elided-line count |
| ripgrep hits | match count, file paths, matched lines (already compact) | beyond a match-count threshold: head/tail window + elided-match count |
| long command log | exit code, capped stderr, head window (invocation), tail window (result/error) | the middle, replaced with an elided-line-count marker |

## 7. Retrieval

The `Retriever` interface and `SqliteFtsRetriever` adapter are implemented
(`src/harness/retrieval/`). The retriever queries the existing FTS5 persistent-
facts index from `harness.memory.facts` without duplicating data across tables.
It takes an open `MemoryStore` (not a path) so it cannot access a database the
caller did not already open.

Retrieval is not yet exposed as an agent tool. The model has no tool to query
persistent facts; adding one is the next step. No vector database is present.

Retrieval is a **tool**, not a distinct protocol action — `request_context` is
deliberately absent from the action vocabulary. It is triggered the same way
any tool call is: the model asks for it when working context lacks what the
current step needs. Its result goes through the same compression and append
pipeline as any other tool result (§6) and lands in the append zone (§2) — it
is fresh, per-step content, so appending it costs the ~0.8 s tail rate, never
a prefix rewrite.

## 8. What this deliberately does not do

- **Timer-based automatic compaction.** DISCOVERY.md recorded a live example:
  an `opencode` process with `compaction: {auto: true, prune: true}` took
  **over six minutes** to answer *"How many files are here? Use a tool,
  answer with a number."* — because automatic compaction rewrote the prefix on
  every step and destroyed the cache each time. Compression here fires only
  from the cost model in §5, never a clock.
- **Loading the entire repository into the prompt.** The stable prefix holds a
  repository *map*, not its contents — a 4-core, no-AVX-512 CPU cannot absorb
  the pp-rate cost of a full source tree, and it would make the prefix itself
  unstable on every file change.
- **A vector database without demonstrated need.** Persistent facts use
  SQLite with optional FTS5. Retrieval is not implemented (§7); embeddings
  are added only if a future benchmark demonstrates a need.
- **Sending the full conversation history verbatim.** Working memory is
  scoped to one run and discarded on completion except its summary
  (ARCHITECTURE.md, Memory). An ever-growing raw transcript would cross the
  pp-rate degradation point (~120 tok/s past 18k tokens, §1) on every step,
  with no second slot available to absorb the cost (§1) — it must be actively
  managed, not simply appended to forever.
