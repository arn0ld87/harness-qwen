# Plan: ready-for-agent Issues loesen (abgeschlossen)

Ziel war: Alle GitHub-Issues mit Label `ready-for-agent` (#5–#22) umsetzen.
Slice-basiert, ein Commit + Arbeitsprotokoll pro Sub-Slice, TDD wo moeglich,
`uv run python -m pytest -m "not local_llm"` als Verifikation.

## Status — abgeschlossen

- v0.2 Hardening (#5–#9, #33): COMPLETE
- v0.3 Runtime & CLI (#10–#16): COMPLETE
- v0.4 Retrieval & Benchmarks (#17–#22): PARTIAL — #17, #18, #19 und #22 abgeschlossen, #20–#21 offen (brauchen das echte Modell)
- #27 `harness benchmark` CLI: abgeschlossen (siehe Phase D)

## Ausfuehrungsprotokoll

### Phase A — v0.2 Hardening
- [#5] WorkspaceBaseline Git-unabhaengig + FileEvidence per Fingerprint
- [#6] ContextOverflow netto: RetrieveAgain bei Overflow skippen, Netto-Token-Rechnung, Hard-Ceiling
- [#7] Shell-Sandbox fail-closed + `--unshare-net` default + doctor-Erkennung
- [#8] Resume UNCERTAIN + Side-Effect-Policy (ToolSpec.idempotency/side_effect)
  - `core.SideEffect` (`none`/`idempotent`/`mutating`), `ToolSpec.side_effect`
    mit fail-closed Default `mutating`. Nicht in `to_openai_tool()`.
  - `StepStatus.UNCERTAIN`; Schema bleibt bei v2 (status hat kein CHECK).
  - Resume differenziert: `model_call` und nicht-mutierende Tools -> `FAILED`,
    mutierende Tool-Steps -> `UNCERTAIN`.
  - Guard: erster identischer Wiederholungsversuch eines UNCERTAIN-Calls wird
    als `uncertain_side_effect` zurueckgemeldet.
  - Bericht: `RunResult.uncertain_steps`, Journal-Event, `open_problems`,
    Append-Hinweis an das Modell.
  - Tests: Crash vor Side Effect, nach Side Effect/vor Checkpoint, nach
    Checkpoint, read-only Tool, Guard-Verhalten.
- [#9] CI: einmaliger Coverage-Lauf, `--cov-fail-under`, Capability-Marker,
  ruff + pytest --cov
  - `scripts/coverage_gate.py`: liest `coverage.json`, prueft globale und
    modulweise Mindestabdeckung, meldet jede Unterschreitung einzeln.
  - Schwellen in `pyproject.toml` (`[tool.coverage_gate]`).
  - Marker `sandbox` fuer bwrap-abhaengige Tests.
  - CI: ein Testlauf mit Coverage statt zwei, `-ra` fuer sichtbare
    Skip-Gruende, Gate als eigener Schritt.
- [#33] Prozess-Substitution klassifizieren, Deckel schliessen
  - `<(...)`/`>(...)` wie `$(...)` als Segment klassifiziert
  - Tiefenlimit endet in CONFIRM statt ALLOW
  - ALLOW traegt die Begruendung der zutreffenden Regel
  - Splitter dedupliziert: classifier importiert shellsplit

### Phase B — v0.3 Runtime & CLI
- [#10] `runtime/` Supervisor (Start/Stop/Health, PID, Attach, Crash,
  stdout secret-safe)
  - `runtime/handle.py`: `RuntimeHandle` mit Ownership (owned/attached)
  - `runtime/argv.py`: Startkommando aus `ModelConfig`/`RuntimeConfig`
  - `runtime/supervisor.py`: start/attach/health/stop. Health-Wait mit Timeout
    und Klassifikation (crashed vs. timeout vs. laedt noch).
  - Graceful stop (SIGTERM), harter Kill nur nach Frist; `stop()` beendet
    niemals einen attached Prozess.
  - stdout/stderr zeilenweise durch `telemetry.redact` in eine Logdatei.
- [#11] Portbesitz, stale/foreign, Startvalidierung (auf #10)
  - `runtime/port.py`: Portbelegung klassifizieren (frei / eigener / fremder
    llama-server / fremder Dienst).
  - Start prueft vorher: belegter Port fuehrt zu klarem Fehler.
  - Nach dem Start wird verifiziert, dass der antwortende Endpoint zum eben
    gestarteten Prozess gehoert.
  - Attach nur bei expliziter Konfiguration.
- [#12] `config/` Schicht (Defaults < File < Env < CLI, redigiert,
  Hardware-Profil, typisiert)
  - `config/schema.py`: `RuntimeConfig`, `ModelConfig`, `SandboxConfig`,
    `HarnessConfig`. Budget bleibt `core.Budget`.
  - Defaults referenzieren bestehende Definitionen, statt Zahlen zu
    duplizieren. Genau ein Eigentuemer pro Wert.
  - `config/resolve.py`: Defaults < Datei < Env (`HARNESS_*`) < CLI, jedes
    Feld mit Herkunft. `ResolvedConfig.origins` traegt den dotted path.
  - Redaktion beim Ausgeben ueber `telemetry.redact`.
  - Hardware-Profil: `config/hardware-profile.json` wird gelesen, wenn
    vorhanden; fehlt es, laeuft alles weiter.
- [#13] `harness run` (Exit-Codes, Resume, Budget/Config-Overrides, kein CoT)
  - `tools/builtin.py`: acht Tools mit ToolSpecs (Risk + SideEffect).
  - `session.py`: eine Stelle weiss, wie die acht Komponenten zusammengehen.
  - `cli_run.py`: Exit-Codes je Stop-Grund, JSON, Resume, Overrides.
  - `--approve-confirmable`: ohne Genehmigungsweg kann ein unbeaufsichtigter
    Lauf nichts tun, was der Classifier nicht ohnehin erlaubt.
  - Systemprompt geschaerft.
- [#14] `harness chat` (gleiche Komponenten, `/status` `/context` `/usage` `/exit`)
  - Kein zweiter Agent-Loop: jeder Turn ist ein Lauf desselben `AgentLoop`.
  - Session-Ziel steht im gecachten Prefix.
  - Nachricht wird in persistierten RuntimeState geschrieben, nicht auf den
    Assembler.
  - Bestaetigungen sind echte Entscheidung eines Menschen.
  - `/status`, `/context`, `/usage`, `/help`, `/exit`.
- [#15] `config show` + `memory inspect` (Provenance, JSON, keine Mutation)
  - `MemoryStore(path, read_only=True)`: `mode=ro`, kein `journal_mode`-Write,
    kein FTS-Aufbau. `migrations.check_schema` prueft die Version, statt sie
    zu heben.
  - `memory/inspect.py`: eine Payload fuer Mensch und `--json`.
  - Redaktion ueber `telemetry.redact.redact_data`.
  - `cli_inspect.py` traegt beide Kommandos; `cli.py` bleibt unter der
    500-Zeilen-Grenze und behaelt `doctor`.
  - `config show`: Wert, Ebene und Quelle je dotted path.
- [#16] `harness doctor` ausbauen (Sandbox/Port/Health/JSON-Exit, auf #7/#10/#12)

### Phase C — v0.4 Retrieval & Benchmarks (PARTIAL)
- [x] #17 `retrieval/` Retriever-Interface + SqliteFtsRetriever (FTS5-Fallback)
- [x] #18 Retrieval als Tool in ToolRegistry (Append-Zone, Prefix-stabil, E2E)
  - `tools/retrieval_tool.py`: async `retrieve_facts(retriever, *, query,
    limit=DEFAULT_LIMIT)` → `ToolResult` mit `render_hits` (source:id-Labels).
    Coroutine dispatcht im Event-Loop-Thread (gleicher Thread wie die
    Kompressionsleiter, kein `to_thread` → SQLite `check_same_thread=False`
    + Store-Lock reichen).
  - `tools/builtin.py`: `RETRIEVE_FACTS`-Spec (Risk.ALLOW, SideEffect.NONE,
    query required + limit optional 1–20). `build_registry(retriever=...)`
    registriert bedingt; nicht in `BUILTIN_SPECS` (ein Run ohne Retriever
    würde sonst ein Tool advertisen, das nicht antworten kann).
  - `session.py build_loop`: baut einen `SqliteFtsRetriever` aus dem Store,
    gibt ihn an `build_registry` (Tool) und an `AgentLoop.retrieve=
    as_retrieve_fn(retriever)` (RetrieveAgain-Rung). Ein Retriever, zwei
    Verbraucher, beide nur appendend.
  - Kriterium 5 (kein Retrieval während ContextOverflow): RetrieveAgain wird
    bei `context_overflow` übersprungen — bereits in `test_context_compressor`
    geprüft, jetzt über `build_loop` aktiviert.
  - Kriterium 6 (stale Resultate markieren): `DropSupersededToolOutputs`
    markiert ältere `retrieve_facts`-Outputs bei Wiederholung (nach Tool-Name).
  - Tests: `tests/test_retrieval_tool.py` (11) — Registration, Shape, E2E
    (retrieve→ToolResult→Answer), Prefix-Hash-Stabilität über Calls.
- [x] #19 `benchmark/` Framework (ID, Fingerprint, Warmup/Mess, JSON, Perzentile)
- [ ] #20 Flag-Sweep (Prozessidentitaet, Invaliditaetsregeln)
- [ ] #21 Harness vs Plain Loop (Task-Suite, Metriken, cold/warm, negative Results)
- [x] #22 Cache-/Prefix-Invarianten (Prefix-Hash pro Call, Cache-Hit-Quote)
  - `benchmark/prefix_invariant.py`: `run_prefix_invariant` treibt echten `PromptAssembler` durch Step-Sequenz, prüft Hash-Stabilität über Appends + Cache-Hit, fängt `PrefixViolation`, reportet Anomalien
  - `benchmark prefix`-Subcommand mit `--steps`, `--probe-undeclared`, `--json`; Exit 0/3/1/2
  - Unit-Tests für PromptAssembler-Vertrag (`test_context_assembler.py`, 11 Tests) und CacheEconomics §5 (`test_cache_economics.py`, 6 Tests) — die Lücken, die der Implementierung vorausgingen
  - `render_cache_report` in `benchmark/report.py`

### Phase D — #27 `harness benchmark` CLI
- [x] #27 CLI für Capability-, Flag- und Task-Benchmarks
  - `cli_benchmark.py`: Typer-Subgruppe `benchmark` mit den klar getrennten
    Unterkommandos `capability`, `flags`, `tasks` (jedes defaultet auf eine
    kanonische Suite unter `benchmarks/`, überschreibbar mit `--suite`) sowie
    `compare` (Vergleich zweier Run-Artefakte).
  - Eine async `build_runner`-Seam konstruiert Provider + Handle über den
    `LlamaServerSupervisor` (`ensure()` startet/attacht und gibt das Handle
    für die Identitätsverifikation); Tests injecten einen Stub-Runner mit
    `FakeProvider` und Schein-Probes (kein echter Server, kein Netzwerk).
  - Config-/Runtime-Profil explizit wählbar: `--config`, `--profile`,
    `--base-url`, `--attach`. Output-Verzeichnis (`--out`) und Run-ID
    (`--run-id`) steuerbar; `--case` wählt Cases aus der Suite.
  - Ungültige Runs (Identität unter dem Run geändert, etc.) werden als
    `INVALID` ausgegeben und führen zu Exit-Code 3, nie zu stillen 0.
  - JSON-Artefakt (`write_run`) plus kompakte Terminal-Zusammenfassung
    (`render_summary`); `--json` gibt den Run als JSON aus.
  - `render_comparison(a, b)` in `benchmark/report.py`: Side-by-side der
    Per-Case-Perzentile, Fingerprint-Kompatibilität (Host/Modell/Runtime)
    und Validität beider Runs.
  - Exit-Codes: 0 success, 3 invalid measurements, 1 execution failure,
    2 bad config.
  - Tests: `tests/test_cli_benchmark.py` (FakeProvider/Stub-Runner),
    `tests/test_benchmark_report.py` (`render_comparison`).

## Naechste Arbeiten

Der naechste Plan ersetzt diesen und definiert die Arbeit an den verbleibenden
offenen Issues (#20–#22, #23–#26, #28–#30) sowie neuen Themen. Bis dahin:
`harness run`, `harness chat` und `harness benchmark` sind der aktuelle Stand,
auf dem alles Weitere aufbaut.