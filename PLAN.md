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
- [x] #8 Resume UNCERTAIN + Side-Effect-Policy (ToolSpec.idempotency/side_effect)

#### Sub-Slice 8.1 — Side-Effect-Klasse und UNCERTAIN-Resume
1. `core.SideEffect` (`none`/`idempotent`/`mutating`), `ToolSpec.side_effect`
   mit fail-closed Default `mutating`. Nicht in `to_openai_tool()` — der
   Prefix bleibt byteidentisch.
2. `StepStatus.UNCERTAIN`; Schema bleibt bei v2 (status hat kein CHECK).
3. Resume differenziert: `model_call` und nicht-mutierende Tools -> `FAILED`
   (sicher wiederholbar), mutierende Tool-Steps -> `UNCERTAIN`.
4. Guard: der erste identische Wiederholungsversuch eines UNCERTAIN-Calls
   wird nicht ausgefuehrt, sondern als `uncertain_side_effect` zurueckgemeldet.
5. Bericht: `RunResult.uncertain_steps`, Journal-Event, `open_problems`,
   Append-Hinweis an das Modell.
6. Tests: Crash vor Side Effect, nach Side Effect/vor Checkpoint, nach
   Checkpoint, read-only Tool, Guard-Verhalten.
- [x] #9 CI: einmaliger Coverage-Lauf, --cov-fail-under, Capability-Marker, ruff+pytest--cov

#### Sub-Slice 9.1 — Einmaliger Lauf und dokumentierte Coverage-Gates
1. `scripts/coverage_gate.py`: liest `coverage.json`, prueft globale und
   modulweise Mindestabdeckung, meldet jede Unterschreitung einzeln.
2. Schwellen in `pyproject.toml` (`[tool.coverage_gate]`) — anhebbar ohne
   Codeaenderung, Startwerte knapp unter dem Ist-Stand statt kuenstlich rot.
3. Marker `sandbox` fuer bwrap-abhaengige Tests, zusaetzlich zum
   bestehenden skipif; `local_llm` und `slow` bleiben unveraendert.
4. CI: ein Testlauf mit Coverage statt zwei, `-ra` fuer sichtbare
   Skip-Gruende, Gate als eigener Schritt mit eindeutiger Fehlermeldung.

### Phase B — v0.3 Runtime & CLI
- [x] #10 runtime/ Supervisor (Start/Stop/Health, PID, Attach, Crash, stdout secret-safe)

#### Sub-Slice 10.1 — Lifecycle eines lokalen llama-server
1. `runtime/handle.py`: `RuntimeHandle` mit Ownership (owned/attached), PID,
   Startzeit, Log-Pfad. Owned und attached sind verschiedene Typzustaende,
   nicht ein Flag, das man vergisst zu pruefen.
2. `runtime/argv.py`: Startkommando aus `ModelConfig`/`RuntimeConfig`, typisiert
   vor `extra_flags`.
3. `runtime/supervisor.py`: start/attach/health/stop. Health-Wait mit Timeout
   und Klassifikation (crashed vs. timeout vs. laedt noch).
4. Graceful stop (SIGTERM), harter Kill nur nach Frist; `stop()` beendet
   niemals einen attached Prozess.
5. stdout/stderr zeilenweise durch `telemetry.redact` in eine Logdatei.
6. Tests: Stub-Server statt 35B-Modell (Start, langsamer Start, Crash beim
   Start, Crash im Betrieb, Secret im Log, Stop-Verweigerung bei attached).
   Zusaetzlich ein `local_llm`-Test, der an die laufende Instanz attached,
   ohne sie zu stoppen.
- [x] #11 Portbesitz, stale/foreign, Startvalidierung (auf #10)

#### Sub-Slice 11.1 — Kein stiller Start gegen einen fremden Prozess
1. `runtime/port.py`: Portbelegung klassifizieren (frei / eigener / fremder
   llama-server / fremder Dienst), inklusive PID und Startzeit des Inhabers.
2. Start prueft vorher: belegter Port fuehrt zu einem klaren Fehler, nicht zu
   einem scheinbar erfolgreichen Start gegen den Altprozess.
3. Nach dem Start wird verifiziert, dass der antwortende Endpoint zum eben
   gestarteten Prozess gehoert (PID/Startzeit, nicht nur "antwortet").
4. Attach nur bei expliziter Konfiguration; stale/foreign wird benannt.
5. Tests: Altserver haelt den Port, neuer Start scheitert, Health bleibt
   erreichbar; fremder Nicht-llama-Dienst; Identitaetspruefung schlaegt fehl.
- [x] #12 config/ Schicht (Defaults<File<Env<CLI, redigiert, Hardware-Profil, typisiert)

#### Sub-Slice 12.1 — Typisierte Konfiguration mit Provenance
1. `config/schema.py`: `RuntimeConfig`, `ModelConfig`, `SandboxConfig`,
   `HarnessConfig`. Budget bleibt `core.Budget` — kein zweites Modell fuer
   dieselben Werte.
2. Defaults referenzieren die bestehenden Definitionen (`budget.py`,
   `llamacpp.py`), statt Zahlen zu duplizieren. Genau ein Eigentuemer pro Wert.
3. `config/resolve.py`: Defaults < Datei < Env (`HARNESS_*`) < CLI, jedes Feld
   mit Herkunft. `ResolvedConfig.origins` traegt den dotted path.
4. Redaktion beim Ausgeben ueber `telemetry.redact`, nicht ueber eine zweite
   Musterliste.
5. Hardware-Profil: `config/hardware-profile.json` wird gelesen, wenn
   vorhanden; fehlt es, laeuft alles weiter (kein Hard-Block fremder Systeme).
6. Tests: Prioritaetskette, Typvalidierung, Redaktion, Provenance, fehlendes
   Profil, kaputte Datei.
- [ ] #13 `harness run` (Exit-Codes, Resume, Budget/Config-Overrides, kein CoT)
- [ ] #14 `harness chat` (gleiche Komponenten, /status /context /usage /exit)
- [x] #15 `config show` + `memory inspect` (Provenance, JSON, keine Mutation)

#### Sub-Slice 15.1 — Inspektion ohne SQLite-Handarbeit
1. `MemoryStore(path, read_only=True)`: Verbindung als `mode=ro`, kein
   `journal_mode`-Write, kein FTS-Aufbau. `migrations.check_schema` prueft
   die Version, statt sie zu heben — Ansehen ist keine Zustimmung zur
   Migration.
2. `memory/inspect.py`: eine Payload fuer Mensch und `--json`, damit beide
   dieselbe Teilmenge sehen. Unlesbare Zeilen werden pro Run als `errors`
   gemeldet, nicht geworfen — ein kaputter Store ist der Anlass des Befehls.
3. Redaktion ueber `telemetry.redact.redact_data` (neu, strukturerhaltend):
   Step-Argumente sind das, womit ein Tool aufgerufen wurde, inklusive
   Auth-Header.
4. `cli_inspect.py` traegt beide Kommandos; `cli.py` bleibt unter der
   500-Zeilen-Grenze und behaelt `doctor`.
5. `config show`: Wert, Ebene und Quelle je dotted path, Secrets doppelt
   entfernt (deklariert per Name, frei formulierte per Textscrubber),
   `ResolvedConfig.warnings` mit ausgegeben.
6. Tests: Provenance, Redaktion in beiden Ausgaben, Run-/Status-Filter,
   unbekannter Run, fehlende Datei, Fremddatei, alte Schema-Version ohne
   Migration, beschaedigter TaskState, Byte-Gleichheit der DB nach dem Lauf.
- [x] #16 `harness doctor` ausbauen (Sandbox/Port/Health/JSON-Exit, auf #7/#10/#12)

#### Sub-Slice 33.1 — Prozess-Substitution und fail-closed Tiefenlimit
- [x] `<(...)`/`>(...)` werden wie `$(...)` als Segment klassifiziert
- [x] Tiefenlimit endet in CONFIRM statt ALLOW
- [x] ALLOW traegt die Begruendung der Regel, die zugetroffen hat
- [x] Splitter dedupliziert: classifier importiert shellsplit

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