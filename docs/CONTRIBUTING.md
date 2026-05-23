# Contributing

Thanks for considering a contribution to Kivski. This document covers the
practical bits: how to set up your dev environment, run the tests, keep
the style checks happy, and submit a PR.

For the *shape* of changes - which module owns what, where to add a new
weapon vs a new actor head vs a new map - read
[`ARCHITECTURE.md`](ARCHITECTURE.md) first.

---

## 1. Setting up the dev environment

We support Python 3.10 - 3.12 (3.11 recommended) and Node 20+.

```powershell
# Clone your fork
git clone https://github.com/<your-user>/kivski-tactical-ai-simulator.git
cd kivski-tactical-ai-simulator

# Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Optional: also install wandb
# pip install -e ".[all]"

# Frontend
npm install
```

On macOS / Linux substitute `source .venv/bin/activate` for the
PowerShell activation line.

If `Activate.ps1` is blocked on Windows, see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md#powershell-execution-policy-blocks-activateps1).

---

## 2. Running the tests

```powershell
# Full suite
pytest

# Only fast tests (skip end-to-end smoke matches)
pytest -k "not slow"

# A single module
pytest tests/unit/test_engine.py -v

# With coverage
pytest --cov=kivski_sim --cov=kivski_agents --cov-report=term-missing
```

Frontend:

```powershell
npm run typecheck   # tsc --noEmit
npm run lint        # eslint
npm run build       # tsc -b && vite build
```

CI runs the unit suite + ruff + tsc + eslint on every push and PR; the
integration suite runs on PRs to `main`.

---

## 3. Code style

| Language   | Tool           | Rules                                                 |
|------------|----------------|-------------------------------------------------------|
| Python     | ruff (lint + format) | Configured in `pyproject.toml`. Line length 110, double quotes, 4-space indent. `select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "TID"]`. |
| Python     | mypy           | `mypy packages tests`. Strict-ish: `warn_unused_configs = true`, `ignore_missing_imports = true`. |
| TypeScript | ESLint         | React + hooks + react-refresh defaults, `--max-warnings 0`. |
| TypeScript | tsc            | `strict: true`. |
| Editor     | EditorConfig   | UTF-8, LF line endings. 4 spaces for Python, 2 for TS/JSON/YAML/Markdown. |

Before opening a PR:

```powershell
ruff check .
ruff format .
mypy packages tests
npm run lint
npm run typecheck
pytest -q
```

If ruff or mypy flag a hot-path issue you really cannot fix in this PR,
add a `# noqa: <rule>` or `# type: ignore[<code>]` with a one-line
comment explaining why, and link an issue tracking the cleanup.

---

## 4. Branching and commit hygiene

- Branch off `main`. Branch names: `feat/<short>`, `fix/<short>`,
  `docs/<short>`, `chore/<short>`.
- Keep commits scoped. We prefer many small commits with focused
  diffs over one giant squash.
- Commit messages: imperative mood, short subject, optional body.

Example:

```
fix(engine): clamp negative reaction-time samples to zero

Without the clamp, a degenerate RNG state could yield a negative
reaction-time sample which fed into combat as a future tick id and
caused a NoneType deref two rounds later.
```

We do **not** require Conventional Commits, but the prefix (`fix`,
`feat`, `docs`, `refactor`, `test`, `chore`) is appreciated.

---

## 5. Opening a PR

1. Push your branch to your fork.
2. Open the PR against `main` on the upstream repo.
3. Fill out the PR template - in particular, describe:
   - What changed.
   - Why (link an issue if applicable).
   - How you tested it.
   - Any breaking changes (schema bump, config field rename, etc.).
4. CI must be green. If it is red, fix it before requesting review;
   reviewers will not look until CI passes.
5. Be patient with reviews. We try to respond within a few days.

### Discuss before big changes

For non-trivial work please open an issue first. "Non-trivial" includes:

- New top-level features (new training algorithm, new game phase, new
  module).
- Schema changes (observation/action space, map JSON, config models,
  replay format).
- Dependency changes.
- Performance work that touches the engine hot path.

A 30-second design sketch in an issue saves everyone a couple of days
of back-and-forth on a PR.

---

## 6. What kind of contributions help most

In rough order:

1. **Bug fixes with a regression test.** Always welcome.
2. **Test coverage for under-tested modules.** Run
   `pytest --cov=...` and look for hotspots.
3. **Map contributions.** Original maps that exercise different
   tactical layouts.
4. **Baseline opponents.** New scripted baselines stress the eval
   suite and give the trainer richer sparring.
5. **Documentation.** If something in the docs was confusing, a PR
   that fixes it is genuinely high-leverage.
6. **Performance.** Profile before and after; submit numbers in the
   PR description.
7. **New RL features.** Opponent modelling, PSRO league, better
   curricula, asymmetric reward shaping. Please open an issue first.

What we are **not** looking for right now:

- Cosmetic renames without a behaviour reason.
- Drive-by dependency bumps in lockfiles only.
- "Improvements" to the prose in the README that change tone without
  changing content.
- Adding copyrighted assets (maps, names, sounds, icons) from any
  commercial game. Originality is a project value, not a nice-to-have.

---

## 7. Adding a new component (quick references)

| Component | See |
|-----------|-----|
| Baseline policy | `README.md` -> "Adding a new baseline" |
| Map             | `MAP_FORMAT.md` |
| Eval scenario   | `README.md` -> "Adding a new evaluation scenario" |
| Weapon          | `ARCHITECTURE.md` -> "Extension points" |
| Actor head      | `ARCHITECTURE.md` -> "Extension points" |
| Phase           | `ARCHITECTURE.md` -> "Extension points" |

---

## 8. Code of conduct

Be kind. Assume good faith. No personal attacks, no harassment, no
exclusionary jokes. Maintainers reserve the right to lock or close
threads / PRs that go off the rails.

We do not have a formal CoC document yet; until we do, the contributor
covenant ([contributor-covenant.org](https://www.contributor-covenant.org/))
is the implicit standard.

---

## 9. License

By contributing you agree that your contributions are licensed under
the [MIT License](../LICENSE).
