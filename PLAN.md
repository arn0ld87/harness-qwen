# Plan: ready-for-agent Issues lösen

Ziel: Alle GitHub-Issues mit Label `ready-for-agent` (#5–#22) umsetzen.
Slice-basiert, ein Commit + Arbeitsprotokoll pro Sub-Slice, TDD wo möglich,
`uv run python -m pytest -m "not local_llm"` als Verifikation.

## Ist-Lage (Assessment-Workflow, 18 Agenten)

- v0.2 Hardening (#5–#9): PARTIAL — Bestandscode, chirurgische Änderungen.
- v0.3 Runtime & CLI (#10–#16): NOT-STARTED außer #16 PARTIAL — neue Pakete.
- v0.4 Retrieval & Benchmarks (#17–#22): NOT-STARTED außer #18 PARTIAL.

## Reihenfolge (nach Abhängigkeiten)

### Phase A — v0.2 Hardening (unabhängig voneinander)
- [x] #5 WorkspaceBaseline Git-unabhängig + FileEvidence per Fingerprint
- [x] #6 ContextOverflow netto: RetrieveAgain bei Overflow skippen, Netto-Token-Rechnung, Hard-Ceiling
- [x] #7 Shell-Sandbox fail-closed + --unshare-net default + doctor-Erkennung
- [ ] #8 Resume UNCERTAIN + Side-Effect-Policy (ToolSpec.idempotency/side_effect)
- [ ] #9 CI: einmaliger Coverage-Lauf, --cov-fail-under, Capability-Marker, ruff+pytest--cov

### Phase B — v0.3 Runtime & CLI
- [ ] #10 runtime/ Supervisor (Start/Stop/Health, PID, Attach, Crash, stdout secret-safe)
- [ ] #11 Portbesitz, stale/foreign, Startvalidierung (auf #10)
- [ ] #12 config/ Schicht (Defaults<File<Env<CLI, redigiert, Hardware-Profil, typisiert)
- [ ] #13 `harness run` (Exit-Codes, Resume, Budget/Config-Overrides, kein CoT)
- [ ] #14 `harness chat` (gleiche Komponenten, /status /context /usage /exit)
- [ ] #15 `config show` + `memory inspect` (Provenance, JSON, keine Mutation)
- [ ] #16 `harness doctor` ausbauen (Sandbox/Port/Health/JSON-Exit, auf #7/#10/#12)

### Phase C — v0.4 Retrieval & Benchmarks
- [ ] #17 retrieval/ Retriever-Interface + SqliteFtsRetriever (FTS5-Fallback)
- [ ] #18 Retrieval als Tool in ToolRegistry (Append-Zone, Prefix-stabil, E2E)
- [ ] #19 benchmark/ Framework (ID, Fingerprint, Warmup/Mess, JSON, Perzentile)
- [ ] #20 Flag-Sweep (Prozessidentität, Invaliditätsregeln)
- [ ] #21 Harness vs Plain Loop (Task-Suite, Metriken, cold/warm, negative Results)
- [ ] #22 Cache-/Prefix-Invarianten (Prefix-Hash pro Call, Cache-Hit-Quote)

## Pro Slice
1. PLAN-Sub-Slice spezifizieren.
2. TDD: Test zuerst (rot), dann Implementierung (grün).
3. `uv run ruff check .` + `uv run python -m pytest -m "not local_llm"`.
4. Commit (imperativ, <50 Zeichen Subject, Issue-Referenz im Body).
5. Issue schließen mit Kommentar + Arbeitsprotokoll-Pointer.