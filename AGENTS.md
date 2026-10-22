# ObjView submission platform continuation

Before continuing platform work, read `HANDOFF.md`. It is the cross-machine
handoff entry point and links to the approved design, implementation plans,
progress, acceptance commands and ordered backlog. Do not rely on an old chat
or machine-local Codex memory to reconstruct project state.

## Constraints to preserve

- Work on `feat/submission-protocol` (or a deliberate successor), not `main`.
  Check `git status` before editing; preserve unrelated user changes.
- The GitHub Pages site remains a separate static site. This repository holds
  the benchmark and development-stage independent API/worker packages.
- Development acceptance checks the workflow with CPU synthetic data and the
  trusted dummy algorithm, not algorithm quality or scientific metric accuracy.
- `objview_submit` must not execute submitted code, invoke Docker, hold its
  socket, or access hidden benchmark data. Its current approval state is only
  a holding state; no scheduler currently consumes it.
- Never treat smoke scores, user-supplied timing, or development credentials as
  official evaluation, trusted runtime measurements, or GitHub OAuth.
- Docker installation was postponed by the user. Recheck the new machine;
  continue Docker-free development where possible. Missing Docker is not a
  reason to weaken isolation or claim real container tests passed.
- Keep secrets, runtime SQLite databases, artifacts and hidden datasets out of
  Git. Never request sudo passwords or change Docker socket permissions to 666.
- Before reporting success, run the relevant tests and record what was actually
  verified. The full suite and benchmark smoke have different scopes.
- Update `docs/superpowers/progress.md` and the relevant plan after each useful
  development checkpoint; refresh `HANDOFF.md` when its current-state/backlog
  facts change. Do not merge, force-push or deploy without authorization.

## Baseline acceptance (repository root)

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest discover -s benchmark/tests -v
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 benchmark/smoke_submission.py
```

At the 2026-10-22 handoff: 84 full-suite tests and 54 benchmark/worker smoke
tests. See `HANDOFF.md` for dependencies, sandbox caveats and remaining work.
These instructions do not require any machine-local skills installation.
