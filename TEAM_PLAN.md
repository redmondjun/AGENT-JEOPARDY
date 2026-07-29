# Agent Jeopardy Team Plan

Status: **Execution contract v1**

Team: **Nandh (full stack), Sara (full stack), Jun (front end/web), Vidula (back end)**

Runtime: **Python 3.12**

Submission: **`agent.zip` with `main.py` at the ZIP root**

This file is the team’s source of truth for architecture, ownership, interfaces,
merge order, testing, and competition operations. The organizer-provided
`README.md`, `main.py`, and `jeopardy.py` remain the source of truth for event
rules and API behavior.

---

## 1. Mission and Non-Negotiable Facts

We are building an unattended Python agent, not a web application.

The agent must:

1. Discover every currently open tile, including every ID in each cell’s
   `open_ids` stack.
2. Work multiple independent tiles concurrently.
3. Give the fixed model tools for real computation, file analysis, and
   stateful HTTP interactions.
4. Keep each tile inside strict turn, token, CPU, and wall-clock budgets.
5. Validate every answer before submission because a wrong scored answer loses
   25% of the tile value and causes a growing lockout.
6. Serialize final submissions around the global one-submission-per-three-second
   team limit.
7. Re-poll the board so the agent stops working on tiles another team claimed.
8. Reschedule incorrect answers after cooldown instead of permanently dropping
   them.
9. Run for the full scored round without human input or a single uncaught tile
   failure crashing the process.
10. Package every imported module inside `agent.zip`.

Confirmed constraints from the starter guide:

- Hosted image: `python:3.12-slim`, 2 CPUs, 2 GB RAM.
- No general outbound or inbound internet; only the event server is reachable.
- Fixed model: Claude Haiku 4.5 through the event proxy.
- `max_tokens` is capped at 4096 per model call.
- Model traffic shares a per-minute token rate limit.
- Dependencies install at submission time from root `requirements.txt`.
- ZIP limit: 20 MB compressed and 200 MB uncompressed.
- `main.py` must be at the ZIP root.
- Practice submissions have no point penalty and a flat 10-second tile cooldown.
- Scored wrong answers have a 30-second initial cooldown that doubles up to
  eight minutes.
- Every API submission outcome is returned as normal JSON, often with HTTP 200.

---

## 2. What We Are Not Building

- No Next.js application.
- No Tailwind UI.
- No PostgreSQL database.
- No microservices, queues, Redis, Docker Compose, or cloud deployment.
- No browser engine unless practice proves `requests` and HTML parsing cannot
  solve the Dark Web category within the hosted image.
- No changes to `jeopardy.py` unless an organizer API defect is demonstrated
  with a regression test.
- No architecture that requires network access beyond the event server.

Every minute should improve solve rate, speed, verification, or unattended
survival.

---

## 3. Target Repository Structure

```text
AGENT-JEOPARDY/
├── main.py                     # Nandh: required hosted entrypoint
├── jeopardy.py                 # Organizer plumbing; treat as read-only
├── contracts.py                # Nandh: frozen cross-team interfaces
├── requirements.txt            # Vidula: dependency owner
├── orchestrator/               # Nandh only
│   ├── __init__.py
│   ├── board_loop.py
│   ├── scheduler.py
│   ├── priority.py
│   ├── submission_gate.py
│   └── state.py
├── solver/                     # Sara only
│   ├── __init__.py
│   ├── agent_loop.py
│   ├── prompts.py
│   ├── registry.py
│   ├── answer_parser.py
│   ├── verification.py
│   └── specialists/
│       ├── data.py
│       ├── documents.py
│       ├── cryptic.py
│       ├── code.py
│       └── optimization.py
├── tools/
│   ├── web/                    # Jun only
│   │   ├── __init__.py
│   │   ├── session.py
│   │   ├── html.py
│   │   ├── forms.py
│   │   └── tool.py
│   └── runtime/                # Vidula only
│       ├── __init__.py
│       ├── files.py
│       ├── python_exec.py
│       ├── archives.py
│       ├── processes.py
│       └── tool.py
├── tests/
│   ├── orchestrator/           # Nandh
│   ├── solver/                 # Sara
│   ├── web/                    # Jun
│   ├── runtime/                # Vidula
│   ├── contract/               # Nandh coordinates; every owner contributes
│   └── fixtures/
├── scripts/                    # Vidula only
│   ├── build_agent.sh
│   └── verify_zip.sh
└── TEAM_PLAN.md
```

Directories are ownership boundaries, not suggestions. A person needing a
change in another owner’s directory requests it through a small issue or pull
request comment rather than editing it directly.

---

## 4. Runtime Architecture

```text
                         EVENT BOARD
                             |
                             v
                 +-------------------------+
                 | board poll + open_ids   |  Nandh
                 | flattening + dedupe     |
                 +------------+------------+
                              |
                              v
                 +-------------------------+
                 | priority scheduler      |  Nandh
                 | points / latency / odds |
                 +------------+------------+
                              |
                 bounded worker pool
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
  +---------------+   +---------------+   +---------------+
  | solver loop   |   | web tools     |   | runtime tools |
  | model + tools |   | HTTP/session  |   | files/Python  |
  | Sara          |   | Jun           |   | Vidula        |
  +-------+-------+   +-------+-------+   +-------+-------+
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                 +-------------------------+
                 | candidate + evidence    |  Sara
                 | format verification     |
                 +------------+------------+
                              |
                              v
                 +-------------------------+
                 | submission gate         |  Nandh
                 | board recheck + limiter |
                 +------------+------------+
                              |
                              v
                        EVENT SUBMIT API
```

### Separation of responsibilities

- The orchestrator chooses **which tile and when**.
- The solver decides **how to solve one tile**.
- Tools perform actions but never choose tiles or submit answers.
- Only the submission gate calls `jp.submit()`.
- Only `main.py` assembles the production system.

---

## 5. Frozen Cross-Team Contracts

Nandh owns `contracts.py`. Contract v1 must be merged before parallel feature
work. Changes require acknowledgement from all four teammates.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

AnswerFormat = Literal["exact", "exact_ci", "numeric", "literal", "validator"]


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    category: str
    points: int
    prompt: str
    answer_format: AnswerFormat
    workdir: Path
    files: tuple[Path, ...]
    deadline_monotonic: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: Mapping[str, Any]
    timeout_seconds: float


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    error_code: str | None = None
    elapsed_ms: int = 0
    exact_value: str | None = None


@dataclass(frozen=True)
class CandidateAnswer:
    value: str
    confidence: float
    evidence: tuple[str, ...]
    strategy: str
    exact_value_from_tool: bool = False


@dataclass(frozen=True)
class SolveResult:
    candidate: CandidateAnswer | None
    retryable: bool
    failure_code: str | None = None
    retry_after_seconds: float | None = None


class Tool(Protocol):
    name: str

    def execute(self, request: ToolRequest, task: TaskContext) -> ToolResult: ...


class TileSolver(Protocol):
    def solve(self, task: TaskContext) -> SolveResult: ...
```

Contract invariants:

- `confidence` is always in `[0.0, 1.0]`.
- Every tool request has a finite timeout.
- Tools return structured errors instead of leaking raw exceptions.
- Tool output has a hard byte/character limit.
- `exact_value` bypasses model retyping and is passed programmatically into the
  candidate answer.
- Solvers never call `jp.submit()`.
- Tool modules never import orchestrator modules.
- All file paths resolve inside the tile’s assigned work directory.

---

## 6. Ownership and Branches

| Person | Role | Branch | Exclusive ownership |
|---|---|---|---|
| **Nandh** | Full stack, integration lead | `feat/nandh-agent-core` | `main.py`, `contracts.py`, `orchestrator/`, `tests/orchestrator/` |
| **Sara** | Full stack, solver lead | `feat/sara-solver-engine` | `solver/`, `tests/solver/`, solver fixtures |
| **Jun** | Front end/web specialist | `feat/jun-stateful-web-tools` | `tools/web/`, `tests/web/`, web fixtures |
| **Vidula** | Back end/runtime lead | `feat/vidula-runtime-release` | `tools/runtime/`, `tests/runtime/`, `scripts/`, `requirements.txt`, CI |

`jeopardy.py` is organizer-owned and read-only. `README.md` is the organizer
guide. Nandh owns final assembly but does not rewrite specialist internals.

---

## 7. Nandh Workstream: Orchestration, Strategy, and Submission

### Mission

Keep the process alive, keep workers busy on valuable tiles, and ensure only
verified, still-open answers reach the API at a legal rate.

### Implementation update from Nandh — 2026-07-29

Status: **Core implementation complete on `feat/nandh-agent-core`; specialist
integration and live practice validation remain.**

Completed:

- Added frozen, dependency-free cross-team contracts in `contracts.py` for
  tasks, candidates, solve results, tools, solvers, and the game API.
- Replaced the serial starter flow with a long-running, dependency-injected
  orchestrator while retaining a bounded naive fallback for local connectivity
  checks.
- Added a thread-safe tile lifecycle with atomic worker claims so one tile
  cannot be scheduled twice.
- Added expected-points-per-second prioritization with deterministic,
  faster-first tie-breaking.
- Added bounded daemon workers and hard orchestration deadlines so a hung tile
  cannot permanently consume a worker slot or prevent process exit.
- Added a verified-answer submission gate with format validation, immediate
  board recheck, and a 3.1-second global submission interval.
- Implemented every documented submission result: correct, incorrect,
  already-claimed, lockout, rate-limit, wrong-phase, forbidden, voided, and
  unknown-task outcomes.
- Preserved verified candidates across rate limits and lockouts instead of
  paying for a second model solve.
- Added scored/practice cooldown scheduling without blocking other workers.
- Added rejected-answer history so deterministic solvers cannot repeat an
  already penalized answer.
- Added exact tool-value pass-through so model transcription cannot corrupt an
  exact-match token.
- Added defensive contract copies, finite timeout validation, solve-attempt
  budgets, worker-exception isolation, and safe low-confidence rejection.
- Added automatic `solver.build_solver()` integration. A missing solver uses
  the safe zero-confidence fallback; an internally broken solver fails loudly.
- Documented all new runtime knobs in `.env.example`.
- Added 39 unit and contract tests. All pass on the hosted runtime equivalent,
  Python 3.12.13.

Validation evidence:

```text
Python 3.12.13
Ran 39 tests in 0.007s
OK
git diff --check: clean
```

### Finale hardening update from Nandh — 2026-07-29

Status: **Integrated, fully tested, and staged for pre-finale deployment.**

Completed:

- Increased bounded concurrency to six category-diverse workers and tuned
  expected-points-per-second ordering so the opening wave covers one fast tile
  from every category instead of clustering workers on one column.
- Added phase-change worker quarantine so stale practice or qualifier work
  cannot consume finale capacity or submit after its board closes.
- Added hard model/tool deadline boundaries, retryable typed failures, and
  task-relative file prompts to prevent hung calls and rejected absolute paths.
- Bounded document reads to preserve model context and added authoritative
  exact-value auto-finalization when the model omits the answer envelope.
- Preserved a valid final answer that arrives on the response crossing the
  token budget; the spent response now passes through the normal verifier
  instead of being discarded.
- Added conservative tool-grounded confidence: normal answers supported by a
  successful `web` or `read_file` result receive `0.82`; ungrounded answers
  remain `0.70` and authoritative exact tool values remain `0.95`.
- Retained the 3.1-second serialized submission interval, strict higher-tier
  gate, and six-worker ceiling after an expected-score and proxy-capacity
  review found no evidence supporting riskier settings.
- Added secret-safe lifecycle, dispatch, tool, candidate, and submission logs
  for rapid live-round diagnosis.

Latest validation evidence:

```text
200 tests passed, 10 subtests passed in 13.23s
git diff --check: clean
qualifier result: 5 correct tiles, 1000 points, 0 penalties
```

Still pending before the integrated agent can score:

- Sara must provide `solver.build_solver()` and the model tool-use loop.
- Jun and Vidula must provide their web and runtime tool implementations.
- CPU-heavy and model-call limits must be enforced inside the respective tool
  and solver implementations in addition to the generic tile-worker bound.
- Run the combined agent against the live practice board with the team key.
- Build and verify the final allowlisted `agent.zip` after all imports merge.

### Deliverables

1. Preserve `main.py` as the required root entrypoint.
2. Create and freeze `contracts.py`.
3. Implement a long-running board loop that:
   - polls the board with jitter and backoff;
   - selects the live board from phase;
   - consumes every `open_ids` entry through `jp.open_tiles()`;
   - subtracts solved, claimed, in-flight, and cooling-down tiles;
   - survives transient server and per-tile errors.
4. Implement bounded concurrency:
   - configurable active tile workers;
   - no more than two CPU-heavy operations simultaneously;
   - bounded model-call concurrency based on measured token throttling;
   - graceful cancellation when a tile is claimed elsewhere.
5. Implement priority scheduling using observed practice data:

   ```text
   expected_value = points × estimated_solve_probability
   priority       = expected_value / estimated_seconds_to_verified_answer
   ```

   Tie-break toward faster, already-calibrated categories during scored rounds.
6. Implement the tile state machine.
7. Implement the submission gate:
   - require a verified candidate;
   - recheck the board immediately before submit;
   - enforce a global submission interval greater than three seconds;
   - handle every documented `result` value;
   - schedule cooldown retries without blocking workers;
   - never resubmit `correct`, `already_claimed`, `voided`, or dead-phase work.
8. Assemble Sara, Jun, and Vidula’s modules in `main.py`.

### Tile state machine

```text
DISCOVERED -> QUEUED -> FETCHING -> SOLVING -> VERIFYING -> READY
     ^          |          |           |           |          |
     |          |          |           |           |          v
     |          |          |           |           |     SUBMITTING
     |          |          |           |           |       /     \
     |          |          |           |           |  CORRECT   INCORRECT
     |          |          |           |           |             |
     |          |          |           |           +-------------+
     |          |          |           |              cooldown
     |          |          |           |
     +----------+----------+-----------+
               retryable failures

Any state -> DEAD when claimed elsewhere, voided, wrong phase, or event ended.
Any state -> FAILED only after explicit budget exhaustion or non-retryable error.
```

### Acceptance tests

- All state transitions are unit tested.
- One worker exception cannot terminate the board loop.
- A tile claimed during solving is never submitted.
- `rate_limited` honors server `retry_in` without sleeping the whole scheduler.
- `incorrect` creates a retry timestamp rather than permanent abandonment.
- The submission gate emits at most one call per configured interval.
- The process remains useful when one category’s solver repeatedly fails.

### Files Nandh must not edit

- `solver/**`
- `tools/web/**`
- `tools/runtime/**`
- `requirements.txt`

---

## 8. Sara Workstream: Solver Engine, Specialization, and Verification

### Mission

Given one normalized task and registered tools, produce an evidence-backed
candidate answer without submitting it.

### Deliverables

1. Implement the Anthropic tool-use loop:
   - category-aware system prompt;
   - tool schemas;
   - `tool_use` execution;
   - `tool_result` feedback;
   - bounded turns and tokens;
   - compact conversation history;
   - explicit final-answer envelope.
2. Implement category routing for all six categories:
   - Needle in the Haystack;
   - The Dark Web, delegating actions to Jun’s tools;
   - Ship It, delegating execution to Vidula’s tools;
   - Ancient Scrolls;
   - Cryptic;
   - Heavy Compute, delegating execution to Vidula’s tools.
3. Implement specialist prompts and deterministic preprocessing:
   - tabular schema/profile summaries for large data;
   - chunk/index/search strategy for long documents;
   - encoding, archive, and binary identification for Cryptic;
   - test-first code diagnosis for Ship It;
   - constraint extraction and independent objective checking for Heavy Compute.
4. Implement answer parsing for all server formats:
   - `exact`;
   - `exact_ci`;
   - `numeric`;
   - `literal`;
   - `validator`.
5. Implement verification that is stronger than “ask the model again”:
   - recompute numeric answers when possible;
   - validate literal parsing locally;
   - check optimization constraints and objective;
   - require code tests or reproducible command evidence;
   - preserve exact tool-emitted tokens without model transcription.
6. Return calibrated confidence plus evidence to Nandh’s submission gate.

### Acceptance tests

- The model can request multiple tools over multiple turns.
- Unknown tool names and malformed arguments return tool errors, not crashes.
- Turn and token caps end with a typed retryable failure.
- Large tool results are summarized or truncated deterministically.
- Exact tokens from tools reach `CandidateAnswer.value` unchanged.
- Literal and numeric answers normalize without semantic change.
- Every category has easy, medium, hard, malformed, and timeout fixtures.
- Prompt changes run against the full practice evaluation set.

### Files Sara must not edit

- `main.py`
- `contracts.py`
- `orchestrator/**`
- `tools/web/**`
- `tools/runtime/**`

---

## 9. Jun Workstream: Stateful HTTP and Web Tasks

### Mission

Solve The Dark Web category through reliable HTTP sessions, cookie retention,
HTML understanding, and multi-step forms. This uses Jun’s web/frontend
expertise without building an unrelated UI.

### Deliverables

1. Build a per-tile `requests.Session` wrapper.
2. Restrict requests to URLs allowed by the task and reachable event host.
3. Support:
   - GET and POST;
   - cookies across steps;
   - redirects;
   - query parameters;
   - form-encoded and JSON bodies;
   - custom non-secret headers;
   - timeout and response-size limits.
4. Parse HTML with BeautifulSoup/lxml into a compact semantic representation:
   - title and visible text;
   - links with resolved URLs;
   - forms, methods, actions, and named controls;
   - hidden inputs and CSRF-style tokens;
   - tables and relevant validation messages.
5. Implement form submission by label/name/value rather than brittle visual
   coordinates.
6. Detect and report:
   - login rejection;
   - expired session;
   - redirect loops;
   - missing controls;
   - non-HTML content;
   - oversized response;
   - server timeout.
7. Redact passwords, cookies, authorization headers, API keys, and hidden
   sensitive fields from logs and model-visible errors.
8. Build a local fixture server covering multi-step login and cookie flows.

### Acceptance tests

- Cookie set on step one is present on step two.
- Hidden form tokens are preserved.
- Relative links and actions resolve correctly.
- Wrong credentials produce a typed rejection rather than an infinite retry.
- Redirect loops terminate within a fixed bound.
- Passwords and cookies never appear in logs or returned tool output.
- A malformed page cannot crash the agent process.
- Two concurrent tiles never share sessions or cookies.

### Files Jun must not edit

- `main.py`
- `contracts.py`
- `orchestrator/**`
- `solver/**`
- `tools/runtime/**`

---

## 10. Vidula Workstream: Runtime Tools, Safety, Tests, and Release

### Mission

Give the solver safe, bounded access to real computation and files, then prove
the exact ZIP will import and run in the hosted environment.

### Deliverables

1. Implement file tools scoped to `TaskContext.workdir`:
   - list metadata;
   - bounded text/binary reads;
   - targeted line/byte ranges;
   - safe scratch-file writes;
   - path traversal rejection.
2. Implement Python/process execution:
   - finite timeout;
   - stdout/stderr capture with size limits;
   - process-group termination on timeout;
   - working directory locked to the tile;
   - no shell interpolation by default;
   - CPU-heavy semaphore capped at two.
3. Implement safe archive inspection/extraction:
   - ZIP/TAR detection;
   - member count and expanded-size limits;
   - absolute path and `..` rejection;
   - nested archive depth limit.
4. Define stable error codes:
   - `INVALID_ARGUMENT`;
   - `NOT_FOUND`;
   - `PATH_BLOCKED`;
   - `TIMEOUT`;
   - `OUTPUT_TOO_LARGE`;
   - `PROCESS_FAILED`;
   - `UNSUPPORTED_FORMAT`;
   - `DEPENDENCY_UNAVAILABLE`.
5. Own dependencies. Prefer hosted packages; add a package only when a
   measured task requires it and submit-time installation is verified.
6. Build CI and local checks for Python 3.12.
7. Build `agent.zip` from an explicit allowlist, never `zip -r .`:
   - include root `main.py`, `jeopardy.py`, `contracts.py`, requirements, and
     imported Python packages;
   - exclude `.git`, `.env`, tests, fixtures, caches, logs, and task data;
   - verify `main.py` is at ZIP root;
   - verify compressed/uncompressed size;
   - scan for credentials;
   - import/compile in a clean temporary directory;
   - print commit SHA and ZIP checksum.

### Acceptance tests

- `../../secret` and absolute paths are rejected.
- Archive traversal cannot write outside workdir.
- Hung child processes and descendants are terminated.
- Output is capped with an explicit truncation marker.
- Concurrent CPU jobs never exceed two.
- Same-session tool calls reuse task scratch data safely.
- A clean ZIP contains every production import and no secret or cache.
- Build output is reproducible from the same commit.

### Files Vidula must not edit

- `main.py`
- `contracts.py`
- `orchestrator/**`
- `solver/**`
- `tools/web/**`

---

## 11. Integration Order

```text
Gate 0: starter committed and Python 3.12 baseline understood
  |
  v
Gate 1: Nandh merges contracts.py + package skeleton
  |
  +------------------+------------------+------------------+
  |                  |                  |                  |
  v                  v                  v                  v
Nandh core       Sara solver       Jun web tools      Vidula runtime
  |                  |                  |                  |
  +------------------+------------------+------------------+
                              |
                              v
Gate 2: contract tests and fake-tool integration
                              |
                              v
Gate 3: main.py production assembly
                              |
                              v
Gate 4: practice matrix, failures become regression tests
                              |
                              v
Gate 5: clean ZIP deploy and unattended canary
```

Recommended merge order:

1. Nandh: contracts and skeleton.
2. Vidula: runtime tools, test harness, build script.
3. Jun: web tools.
4. Sara: solver engine and specialists.
5. Nandh: scheduler, submission gate, and production assembly.
6. Vidula: final clean-ZIP verification.

Parallel work starts immediately after Gate 1.

---

## 12. Git Rules That Prevent Collisions

Each person creates only their assigned branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c <branch-from-section-6>
```

Before opening or updating a pull request:

```bash
git fetch origin
git rebase origin/main
```

Rules:

- Never commit directly to another person’s directory.
- Never use `git add -A` in a mixed worktree; stage explicit owned paths.
- One behavior change per pull request.
- No secrets, `.env`, downloaded task materials, or generated ZIP files.
- All PRs include tests and the command/output used to verify them.
- Contract changes require all four teammates to acknowledge the interface.
- Nandh reviews integration behavior; the directory owner reviews specialist
  correctness.
- Vidula reviews every dependency or release-script change.
- Rebase before merge; do not force-push `main`.

Pull request prefixes:

- `core:` Nandh
- `solver:` Sara
- `web:` Jun
- `runtime:` Vidula
- `contract:` all four reviewers
- `release:` Vidula plus Nandh

---

## 13. Test and Evaluation Strategy

### Test layers

```text
                    Hosted unattended practice run
                              /       \
                      Practice matrix  ZIP canary
                            /             \
                    End-to-end contract tests
                  /        |        |         \
             core unit  solver eval web fixture runtime safety
```

### Required contract tests

1. The scheduler gives a normalized `TaskContext` to a fake solver.
2. The solver calls a fake tool and consumes its `ToolResult`.
3. Jun and Vidula’s real tools pass the same tool contract suite.
4. A tool-emitted exact token reaches submission unchanged.
5. A malformed answer never reaches `jp.submit()`.
6. A tile claimed between solve and submit is discarded.
7. A rate-limited submission is rescheduled correctly.
8. A wrong practice answer becomes eligible after ten seconds.
9. One worker crash does not stop other workers or the board loop.
10. ZIP verification catches a missing imported module.

### Practice matrix

Track every category and tier:

| Category | 100 | 200 | 300 | 400 | 500 | Owner |
|---|---:|---:|---:|---:|---:|---|
| Needle in the Haystack | | | | | | Sara |
| The Dark Web | | | | | | Jun + Sara |
| Ship It | | | | | | Vidula + Sara |
| Ancient Scrolls | | | | | | Sara |
| Cryptic | | | | | | Sara + Vidula |
| Heavy Compute | | | | | | Vidula + Sara |

For every attempt record:

```text
task_id, category, tier, strategy, solved, elapsed_seconds,
model_calls, tool_calls, answer_format, failure_code, git_sha
```

No strategy is “done” because it solved one seed. Require both practice variants
in a cell or repeated runs against independent fixtures.

### Regression rule

Every observed practice or scored failure is reduced to a fixture and test
before the fix merges. Otherwise later prompt and concurrency changes will
reintroduce it silently.

---

## 14. Submission Policy

The submission gate applies checks in this order:

```text
candidate exists?
  -> confidence above category/tier threshold?
    -> answer parses under answer_format?
      -> evidence/constraint checks pass?
        -> tile still open and live?
          -> global submission slot available?
            -> submit exact programmatic value
```

Handling API results:

| Result | Action |
|---|---|
| `correct` | Mark solved permanently |
| `incorrect` | Apply penalty record; schedule after returned/known cooldown |
| `already_claimed` | Mark dead; cancel duplicate work |
| `locked_out` | Schedule at server retry time |
| `rate_limited` | Honor `retry_in`; keep candidate if tile stays open |
| `wrong_phase` | Refresh board; mark phase-stale work dead |
| `forbidden` | Fatal deployment/config signal during scored round |
| `voided` | Mark dead with no retry |
| `unknown_task` | Mark dead and log contract mismatch |
| missing `result` | Treat as API/auth failure, never as an incorrect answer |

Confidence must be calibrated from practice. Do not use one universal guessed
threshold for all categories and tiers.

---

## 15. Competition Timeline

The official schedule is short, so integration must happen early.

### T+0:00 to T+0:20

- All four clone and run syntax/import checks.
- Create local `.env`; never commit it.
- Confirm one teammate securely holds the team key backup.
- Run the untouched baseline only to validate environment/API connectivity.
- Nandh merges contract v1 and package skeleton.

### T+0:20 to T+0:55

- Four workstreams implement their smallest useful vertical slice.
- Vidula gets build/ZIP verification working immediately.
- Jun targets a 100-level Dark Web practice fixture.
- Sara targets one low-tier data/document fixture with the tool loop.
- Nandh gets a long-running scheduler working with fake solver results.

### T+0:55 to T+1:15

- Merge runtime, web, solver, then core integration.
- Run one practice tile from every category.
- Turn failures into owner-specific issues, not group edits.

### T+1:15 to T+1:25

- Freeze features.
- Build from a clean checkout.
- Inspect ZIP contents and checksum.
- Deploy before Round 1.
- Confirm hosted status and logs.

### Round 1, T+1:30

- Treat it as real scoring: 30% of all available points.
- One person watches logs and classifies failures; nobody edits the live branch
  without a reproducible issue.

### T+1:45 to T+2:30

- Prioritize Round 1 failure classes by lost points and recurrence.
- Add regression tests before fixes.
- Recalibrate concurrency, confidence, and category priority.
- Deploy at least one canary well before the Finale.

### T+2:30 to T+2:40

- Final feature freeze and clean ZIP verification.
- Retain the last known-good ZIP and commit SHA for rollback.

### Finale, T+2:45

- Start wide immediately because every stack entry is open.
- Monitor only; make a hotfix only for a repeated high-blast-radius failure.

---

## 16. Failure Modes and Owners

| Failure | Prevention/recovery | Owner |
|---|---|---|
| Agent exits on one bad tile | Per-tile exception boundary and supervisor loop | Nandh |
| Agent works only card faces | Always use all flattened `open_ids` | Nandh |
| Serial agent loses board width | Bounded concurrent scheduler | Nandh |
| Model loops forever | Turn, token, and wall-clock caps | Sara |
| Model retypes an exact token incorrectly | `ToolResult.exact_value` pass-through | Sara + Nandh |
| Wrong answer submitted confidently | Deterministic verification and calibrated gate | Sara + Nandh |
| Web login loses cookies | Per-tile persistent session | Jun |
| Two web tasks leak sessions | Session isolation keyed by task ID | Jun |
| HTML/prompt tries to exfiltrate secrets | Redaction and untrusted-content boundary | Jun + Sara |
| Python process hangs | Timeout plus process-group kill | Vidula |
| Archive writes outside workdir | Safe extraction and path validation | Vidula |
| Huge output consumes model context | Bounded output and targeted reads | Vidula |
| Too much concurrency throttles model | Measured model-call semaphore | Nandh |
| Submission rate limit wastes candidates | Serialized submission gate | Nandh |
| Wrong answer is never retried | Cooldown-aware state machine | Nandh |
| ZIP misses an import | Clean-directory import smoke test | Vidula |
| Secret enters Git or ZIP | `.gitignore`, content scan, explicit allowlist | Vidula |
| Teammates overwrite each other | Exclusive directories and contract-first work | All |

---

## 17. First Pull Request for Each Person

### Nandh

Title: `contract: add agent interfaces and orchestration skeleton`

Includes only:

- `contracts.py`;
- `orchestrator/__init__.py`;
- placeholder interfaces/state definitions;
- contract/state tests.

### Sara

Title: `solver: add bounded tool-use loop and answer parser`

Includes only:

- tool-use loop;
- fake tool adapter;
- answer-format parsing;
- loop-limit and exact-value tests.

### Jun

Title: `web: add isolated HTTP sessions and semantic form parsing`

Includes only:

- session wrapper;
- HTML/form parser;
- local fixture server;
- cookie, hidden-token, redirect, and redaction tests.

### Vidula

Title: `runtime: add safe Python execution and submission build checks`

Includes only:

- bounded process runner;
- file containment;
- runtime tests;
- explicit ZIP build/verification scripts.

---

## 18. Definition of Done

A pull request is done only when:

- It stays inside the owner’s files or has an approved contract change.
- Happy path, edge cases, timeout, malformed input, and error path are tested.
- Tests run on Python 3.12.
- No secret or downloaded team-specific task material is present.
- Every loop, subprocess, HTTP call, and output has a bound.
- Logs contain task ID and error code without credentials.
- The implementation behaves correctly under concurrent tasks.
- The PR states how it was verified and shows the result.
- Documentation remains consistent with the implementation.

The integrated agent is competition-ready only when:

- All six categories have at least one proven practice solve.
- At least one full unattended practice cycle completes without a process exit.
- Wrong answers are rare, explained, and become regression tests.
- The exact ZIP passes clean-directory compilation/import checks.
- Hosted logs confirm the deployed agent remains alive and polling.
- The team has a known-good rollback ZIP and commit SHA.

---

## 19. Immediate Team Actions

1. Everyone pulls `main` and reads the organizer `README.md` completely.
2. Nandh opens and merges the contract/skeleton PR first.
3. After contract v1 lands, all four branches proceed in parallel.
4. Vidula creates the first safe ZIP build before feature code grows.
5. The team records practice outcomes by category and tier from the first run.
6. Deploy a minimal tool-using agent early; do not wait for all specialists.

If ownership or an interface changes, update this file in the same pull request.
