# Vision

## The premise

A Qwen3.6-35B-A3B running on a 2017 quad-core with a 6 GB GTX 1060 generates
**16.5 tokens per second**. That number is not going to improve much. It is the
hard floor this project is built on.

The bet is that a model this size can still do useful agentic work — if the
runtime around it stops wasting its output. Most of what a coding agent asks a
model to do is not reasoning. It is bookkeeping: remembering the task, tracking
which files were read, recalling that a test failed two steps ago, deciding what
to keep in context. Every token spent on bookkeeping is a token not spent on the
decision, and at 16.5 tok/s the difference is measured in minutes.

So the split is deliberate:

```
the model decides    →    the harness remembers, retrieves, executes, verifies
```

## What this is not

**Not a chatbot.** There is no conversational mode as a goal in itself.

**Not another loop around an OpenAI-compatible endpoint.** That is roughly 200
lines and provides nothing the endpoint doesn't already have. The value has to
come from the parts that a bare loop does not have: hardware-aware runtime
control, measured model capability profiling, cache-aware context management,
enforced structured actions, tool-output compression, persistent task state,
evidence-based verification, and recovery.

**Not a framework.** Single user, single machine, single model family. No
plugin architecture, no service mesh, no abstraction layer waiting for a second
implementation that will never arrive.

**Not optimistic about the model.** Nothing about tool calling, JSON validity,
schema compliance, or multi-step coherence is assumed. Everything is measured
first, in `benchmarks/model-capabilities.json`, and the protocol adapts to what
the measurements show.

## The three findings that shape the design

Discovery produced three facts that are not obvious and that most agent
frameworks get wrong on this hardware. They are documented in full in
[docs/DISCOVERY.md](docs/DISCOVERY.md).

### 1. Appending is free. Rewriting is not.

The prompt cache reuses a stable prefix at **~50× speedup** — a byte-identical
4816-token prompt reprocesses in 0.49 s instead of 24.83 s, and adding tokens to
the end costs 0.82 s.

Change one byte near the front and the entire prefix is recomputed: 25 s at
4.8k tokens, ~106 s at 16k. At 16.5 tok/s, an 8k reprocess is worth about **825
discarded output tokens**.

This makes "compress the context when it gets large" actively harmful when
applied naively. A locally running agent observed during discovery took **over
six minutes** to answer *"How many files are here?"* — because its automatic
compaction rewrote the prefix on every step.

The harness therefore treats the prompt as **append-only by default**, keeps the
prefix byte-stable across an entire run including role switches, and compresses
only when a cost model says the saving amortises over the remaining steps.

### 2. A second concurrent request halves throughput.

Measured: 16.5 tok/s alone, 8.55 tok/s with a second slot active. On one GPU
with one model, parallel agents are queueing with extra bookkeeping.

Sub-agents exist here as **roles executed sequentially** — planner, coder,
tester, reviewer — sharing one model, one slot, and one warm cache. Not as
concurrent processes.

### 3. The model is a hybrid, and its fast path is currently off.

30 of 40 layers are recurrent Gated DeltaNet; only 10 use dense attention. The
running configuration disables the fused GDN kernel outright, and discards ~900
MB of MTP weights it loads and never uses.

A harness that writes a `performance-profile.yaml` nobody applies is decoration.
This one owns the server process, so measured flags actually take effect.

## What "done" means

Version 1 is not finished when the code runs. It is finished when it has been
**measured against the alternative** and won — higher task success rate, fewer
tokens, fewer retries, less hallucination — on the reproducible tasks in
`benchmarks/`.

If it does not beat a plain prompt loop on those tasks, it has no reason to
exist, and the benchmark should say so plainly rather than be quietly adjusted
until it agrees.

## Ordering principle

When two designs compete:

```
1. reliability
2. quality of agent outcomes
3. RAM / VRAM footprint
4. latency
5. simplicity
6. extensibility
7. theoretical maximum features
```

Seven loses to one every time.
