# Troubleshooting

Common issues, mostly Windows-flavoured but with cross-platform notes
where it matters.

If your problem is not here, please open an issue with:

- OS + Python + Node versions
- The full command you ran
- The full traceback (not just the last line)
- Anything you have already tried

---

## Install

### PowerShell execution policy blocks `Activate.ps1`

Symptom (on a fresh Windows install):

```
.\.venv\Scripts\Activate.ps1 : File ... cannot be loaded because running
scripts is disabled on this system.
```

Fix - run **once** as the current user (no admin needed):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Reopen the shell and `Activate.ps1` will work. Alternative one-shot:

```powershell
powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
```

### `pip install -e ".[dev]"` fails on a `pip` <22 venv

Symptom: `ERROR: File "setup.py" not found.` Older pip cannot read
modern `pyproject.toml`.

Fix:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### `numba` build error on Python 3.13

We pin Python to `>=3.10, <3.13`. Numba does not (yet) publish wheels
for 3.13 at the time of writing. Use Python 3.11 or 3.12.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### `torch` install pulls CPU build but you wanted CUDA

`pip install -e ".[dev]"` installs whatever `torch` wheel pip picks
(usually CPU). To switch to a CUDA build, install torch separately
before installing this package:

```powershell
# Example: CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### `ImportError: pettingzoo`

You either skipped the editable install or used a different interpreter
than the one in your venv.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -c "import pettingzoo; print(pettingzoo.__version__)"
```

If that still fails, check `where python` (PowerShell: `Get-Command
python | Select-Object Source`) - the first hit must be inside
`.venv\Scripts\`.

---

## Training

### Smoke run hangs at "compiling Numba"

First import of `kivski_sim` JITs hot paths. On a cold machine this can
take 30 - 60 s. Subsequent runs are cached under `~/.numba_cache`.

If you see ridiculous wait times (> 5 minutes) clear the cache and try
again:

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\numba_cache" -ErrorAction SilentlyContinue
```

### `RuntimeError: CUDA out of memory`

Lower `training.num_envs` or the model size:

```powershell
kivski-train --num-envs 4 --device cuda
# or
kivski-train --num-envs 16 --device cpu
```

Each env holds a small replay buffer; the dominant memory cost is
batched forward / backward inside the PPO update.

If you really need to keep `num_envs` high, reduce `ml.hidden_size`
(256 -> 128) and / or `ml.comm_embedding_dim` (64 -> 32) in the config.

### Training reward plateaus near zero

A few common causes, ordered by likelihood:

1. **Shaping is off and the policy has not learned the sparse signal
   yet.** Enable `reward_shaping.enabled: true` and re-run.
2. **Comm-gate has collapsed.** Check `mean(g_i)` in the telemetry. If
   it is near 0 (channel always closed) or near 1 (channel always
   open), the gate is not learning. Try `gumbel_temperature: 0.5` or
   adjust the gate-penalty `beta`.
3. **Entropy collapse.** If any head's entropy drops to ~0 nats in the
   first 5k episodes, raise `entropy_coef`.
4. **Bad initial seed.** Run with two more seeds before assuming
   anything; PPO has a wide success/failure distribution at small scale.

### `Subprocess died during rollout`

We do not yet use subprocess vec env (it is on the roadmap). If you see
this error it is a stale traceback from a different framework being
imported alongside ours. Run `pip list | findstr stable-baselines` and
remove conflicting installs.

---

## API server

### `kivski-serve` exits immediately with no error

You likely hit a port conflict on 8000. Run on another port:

```powershell
kivski-serve --port 8123
```

In the viewer, edit `apps/web/vite.config.ts` (or the proxy section in
the dev server) to point at the new port.

### WebSocket disconnects in the browser

Symptom in the browser console:

```
WebSocket connection to 'ws://127.0.0.1:8000/ws/match/...' failed
```

Top suspects:

1. The API server is not running. Open <http://127.0.0.1:8000/api/health>
   in a browser tab.
2. CORS allowlist is too strict. The viewer dev origin is
   `http://localhost:5173`. If you changed it, set the env var:

   ```powershell
   $env:KIVSKI_CORS_ORIGINS = "http://localhost:5174,http://127.0.0.1:5174"
   kivski-serve
   ```

3. A proxy / VPN is rewriting the upgrade headers. Test directly with
   `wscat`:

   ```powershell
   npx wscat -c ws://127.0.0.1:8000/ws/match/test
   ```

### `wandb: command not found` / `ImportError: wandb`

W&B is an optional dependency. Install it explicitly:

```powershell
pip install -e ".[wandb]"
```

Then set the credentials:

```powershell
$env:WANDB_API_KEY = "your-key"
$env:KIVSKI_WANDB_PROJECT = "kivski-tactical-ai"
$env:KIVSKI_WANDB_MODE = "online"   # or "offline" for air-gapped
```

If you never want W&B, keep `WANDB_API_KEY` unset and we fall back to
CSV + TensorBoard logging automatically.

---

## Frontend

### `npm install` is extremely slow on Windows

`node_modules` on Windows is notoriously slow. Two mitigations:

1. Make sure the repo lives on an NTFS drive (not a network share).
2. Exclude the repo from Windows Defender real-time scanning - it can
   triple install time.

### `Vite proxy error: ECONNREFUSED 127.0.0.1:8000`

The frontend dev server proxies `/api/*` and `/ws/*` to the Python
backend. If the backend is not running, every fetch is going to fail
with a proxy error - this is expected. Start the backend
(`kivski-serve`) and reload.

### Map shows a placeholder instead of dustline

The frontend ships with a built-in placeholder map (a single empty
rectangle). If you see it, the fetch to `/api/maps/dustline` failed.
Same likely causes as the WebSocket section above: API not running, or
CORS misconfigured.

### `tsc` errors after pulling

```powershell
npm install
npm run typecheck
```

If errors persist, blow away the cache:

```powershell
Remove-Item -Recurse -Force apps/web/node_modules, apps/web/dist
npm install
```

---

## Replays

### `Replay version mismatch`

Replays are tagged with the engine schema version. If you bump the
engine in an incompatible way, older replays cannot be played back. The
fix is to re-record (sorry).

### Replay diverges from training run

Determinism is end-to-end on **CPU**. On CUDA, PyTorch operations can be
non-deterministic depending on cudnn version and benchmark mode. If you
need bit-exact replay of a CUDA-trained policy, force CPU inference:

```powershell
kivski-eval run path/to/checkpoint.pt random --matches 1 --device cpu
```

---

## Tests

### `pytest` collects 0 tests

You probably ran from a sub-directory. Run from the repo root:

```powershell
cd C:\path\to\kivski-tactical-ai-simulator
pytest
```

### `pytest` complains about `asyncio_mode`

You are using an old `pytest-asyncio`. Upgrade:

```powershell
pip install --upgrade "pytest-asyncio>=0.23"
```

### Slow tests are slow

`pytest -k "not slow"` skips end-to-end smoke matches. CI runs the full
suite on every PR but you do not have to locally.

---

## Still stuck?

Open an issue with the template above. If you suspect a determinism
bug, include the seed and the exact command line - those bugs are nasty
without reproducer info.
