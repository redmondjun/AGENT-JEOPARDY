# Competitive Agent Jeopardy v1

Branch: `codex/competitive-agent-v1`

Worktree: `/Users/junlee/workspace/AGENT-JEOPARDY-competitive-agent`

## Status

- [x] Create the isolated branch and worktree.
- [x] Implement task-scoped file, Python, and stateful HTTP tools.
- [x] Implement the Anthropic tool-use loop and adaptive verification.
- [x] Implement board polling, scheduling, cooldowns, and submission gating.
- [x] Add offline unit and integration tests.
- [x] Add configuration documentation and deterministic packaging.
- [x] Pass Python 3.12 tests and package validation.
- [ ] Configure local credentials for practice.
- [ ] Solve at least one practice tile in every category.
- [ ] Solve at least four of six selected 400/500-point practice tiles.
- [ ] Complete a 30-minute practice soak without crashes, duplicate
  submissions, dropped cooldown retries, or submission-rate loops.

## Runtime Contract

- `main.py` owns the long-running scheduler and the only submission path.
- `solver.py` owns model conversations, answer validation, and review.
- `tools.py` owns task-scoped tool execution and never submits.
- `jeopardy.py` remains API-compatible while using thread-local sessions.
- Every ID in `open_ids` is independently schedulable.
- Submissions are separated by at least 3.2 monotonic seconds.
- Incorrect candidates remain eligible after their cooldown and are never
  blindly resubmitted.
- Exact values may flow from tool output to submission through immutable
  answer references without model retyping.

## Defaults

| Setting | Default |
|---|---:|
| `WORKERS` | 6 |
| `MAX_TURNS` | 8 |
| `TILE_TIMEOUT_SECONDS` | 120 |
| `POLL_SECONDS` | 2 |
| `MAX_TILES` | 0 (unlimited) |

Practice accepts format-valid candidates at confidence `0.55`. Scored rounds
accept programmatically captured and deterministically validated candidates at
`0.75`; other candidates require `0.85` and independent reviewer approval.

## Practice Results

No live practice has been run. Credentials are intentionally absent from the
repository. Record only aggregate category/tier success counts and latency
here; never record prompts, answers, or secrets.

Offline verification: 14 tests pass on Python 3.12.13. `agent.zip` contains
the six allowlisted root files and passes CRC validation.

| Category | Tier | Attempts | Correct | Median seconds | Notes |
|---|---:|---:|---:|---:|---|
| Pending | - | 0 | 0 | - | Awaiting local `.env` |

## Acceptance Commands

```bash
python -m unittest discover -s tests -v
python package_agent.py
unzip -l agent.zip
```
