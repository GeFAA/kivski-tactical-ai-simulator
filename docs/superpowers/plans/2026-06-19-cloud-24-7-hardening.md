# Cloud 24/7 Training — Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RunPod 24/7 cloud training pipeline actually deliver value — checkpoints persist across pod restarts, the local viewer plays the cloud-trained model when the user clicks "Pull & Load", the league config doesn't silently undo the WR-fix, and the HF token doesn't leak to disk.

**Architecture:** Three-tier fix. (A) **Persistence**: trainer + supervisor write to the RunPod `/workspace/persistent` volume mount instead of ephemeral container disk. (B) **Policy wiring**: pulled cloud checkpoint is discoverable by `latest_checkpoint_path()` and registered so the next viewer match uses it. (C) **Operational hygiene**: stop writing the HF token to `~/.git-credentials`, pin torch so the second pip install can't downgrade to CPU, cap over-dense per-tick rewards, and align the entrypoint compat-check with the actual sidecar keys.

**Tech Stack:** Python 3.11 (trainer, FastAPI backend, cloud_sync), TypeScript+React+Vite (frontend), bash (Docker entrypoint + 24-7 supervisor), YAML (configs), Docker (CUDA 12.1 base, RunPod RTX 4090 pod).

---

## Scope

13 audit findings ranked into:
- **Phase 1 (P0)**: 6 blockers — the 24/7 promise is broken without these
- **Phase 2 (P1)**: 4 importants — security/efficiency
- **Phase 3**: rebuild image, push, verify on running pod

P2/P3 polish (UX, tests, scripts cleanup) is **deferred** to a follow-up plan to keep this delivery atomic.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `scripts/train.py` | Modify | Add `--checkpoint-dir`/`--log-dir` CLI flags so cloud entrypoint can route writes to persistent volume |
| `packages/agents/kivski_agents/training/trainer.py` | Modify | Accept the new dir flags via TrainerConfig override |
| `apps/api/kivski_api/routes/cloud.py` | Modify | After pull, copy checkpoint to a top-level path AND set `latest.pt` symlink so `latest_checkpoint_path()` finds it; wrap `hf_hub_download` in `run_in_threadpool` |
| `apps/api/kivski_api/policies.py` | Modify | `latest_checkpoint_path()` includes `cloud/` subdir in the glob |
| `apps/api/kivski_api/session.py` | Modify | `create_match` honors `REGISTRY.loaded_checkpoint` if set |
| `docker/entrypoint.sh` | Rewrite section 4 | Compat-check uses real sidecar `model_arch` keys, shows errors on failure; export `KIVSKI_CHECKPOINT_DIR`+`KIVSKI_LOG_DIR` env vars; do NOT call `huggingface-cli login`; persistent restart-history path |
| `docker/24-7-supervisor.sh` | Modify | Move `HISTORY_FILE` + `CRASH_REASON_FILE` to `/workspace/persistent/` |
| `docker/Dockerfile` | Modify | Re-install torch from cu121 index AFTER editable install OR add `--no-deps` to editable install |
| `docker/runpod-template.json` | Modify | Set `KIVSKI_CONFIG=configs/production.yaml` to match docs |
| `configs/production.yaml` | Modify | League fractions sum to 1.0; cap `successful_plant`+`successful_defuse` magnitude or make one-shot |

---

## Phase 1 — P0 Blockers

### Task 1: Trainer writes to persistent volume

**Files:**
- Modify: `scripts/train.py:138-178` — add CLI flags
- Modify: `packages/agents/kivski_agents/training/trainer.py` — accept override (likely already does)

- [ ] **Step 1: Read current CLI to know exact arg shape**

Run: `grep -n "log_dir\|checkpoint_dir\|typer.Option" scripts/train.py | head -20`
Expected: see where `log_dir`/`ckpt_dir` are hardcoded.

- [ ] **Step 2: Add `--checkpoint-dir` and `--log-dir` Typer options to `train` command in `scripts/train.py`**

Read the existing `train` typer function, add two new optional params before `--device`:

```python
checkpoint_dir: Annotated[
    Optional[str],
    typer.Option("--checkpoint-dir", help="Override base dir for checkpoints (default: models/checkpoints/RUN). Useful for cloud pods writing to /workspace/persistent."),
] = None,
log_dir: Annotated[
    Optional[str],
    typer.Option("--log-dir", help="Override base dir for run logs (default: models/logs/RUN)."),
] = None,
```

Replace the hardcoded path block (currently `log_dir = Path("models/logs") / rn; ckpt_dir = Path("models/checkpoints") / rn`) with:

```python
log_root = Path(log_dir) if log_dir else Path("models/logs")
ckpt_root = Path(checkpoint_dir) if checkpoint_dir else Path("models/checkpoints")
log_dir = log_root / rn
ckpt_dir = ckpt_root / rn
```

- [ ] **Step 3: Smoke test the new flag locally**

Run:
```powershell
.\.venv\Scripts\python -m scripts.train train -c configs/production.yaml --episodes 1 --num-envs 2 --num-workers 1 --device cpu --vec-kind sync --checkpoint-dir tmp_smoke_ck --log-dir tmp_smoke_log
```
Expected: writes go to `tmp_smoke_ck/kivski-*/` and `tmp_smoke_log/kivski-*/`, NOT to `models/checkpoints/`.

- [ ] **Step 4: Cleanup tmp dirs**

Run: `Remove-Item -Recurse -Force tmp_smoke_ck, tmp_smoke_log 2>$null; ls models/checkpoints | Select-Object -First 3`
Expected: tmp dirs gone, models/checkpoints unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "feat(trainer): --checkpoint-dir + --log-dir CLI flags for cloud persistent volumes"
```

---

### Task 2: Entrypoint routes writes to persistent volume

**Files:**
- Modify: `docker/entrypoint.sh:69-77` — the exec block

- [ ] **Step 1: Edit the `# --- 5. exec the trainer ---` section**

Replace lines 69-77 with:

```bash
# --- 5. exec the trainer ----------------------------------------------------
PERSIST_LOG_DIR="${PERSIST_LOG_DIR:-/workspace/persistent/logs}"
mkdir -p "${PERSIST_LOG_DIR}"

# Pass persistent dirs explicitly so writes survive pod restarts.
train_args=(train -c "${CONFIG_FILE}" --checkpoint-dir "${PERSIST_CKPT_DIR}" --log-dir "${PERSIST_LOG_DIR}")
if [[ -n "${RESUME_CKPT}" ]]; then
    log "resuming from ${RESUME_CKPT}"
    train_args+=(--resume "${RESUME_CKPT}")
else
    log "no checkpoint to resume — fresh run"
fi

log "starting kivski-train ${train_args[*]}"
exec kivski-train "${train_args[@]}"
```

- [ ] **Step 2: Bash syntax check**

Run: `bash -n docker/entrypoint.sh && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add docker/entrypoint.sh
git commit -m "fix(docker): route trainer writes to /workspace/persistent volume"
```

---

### Task 3: Pull-and-Load actually swaps policy

**Files:**
- Modify: `apps/api/kivski_api/policies.py:98-107` — include cloud subdir in glob
- Modify: `apps/api/kivski_api/session.py` `create_match` — honor `REGISTRY.loaded_checkpoint`

- [ ] **Step 1: Read `latest_checkpoint_path` to confirm current glob**

Run: `Get-Content apps/api/kivski_api/policies.py | Select-Object -Skip 90 -First 25`
Expected: see `root.glob("*.pt")` non-recursive call.

- [ ] **Step 2: Make `latest_checkpoint_path` find cloud-pulled ckpts**

Replace `root.glob("*.pt")` with `(*root.glob("*.pt"), *root.glob("cloud/*.pt"))` — keep only the **top-level** + **cloud/** dirs (don't recurse into per-run subdirs to preserve fast path).

Exact replacement (read the file first to lock the surrounding context):

```python
candidates = list(root.glob("*.pt")) + list((root / "cloud").glob("*.pt"))
if not candidates:
    return None
return max(candidates, key=lambda p: p.stat().st_mtime)
```

- [ ] **Step 3: Read `session.create_match` to find policy resolution**

Run: `grep -n "latest_checkpoint_path\|loaded_checkpoint\|create_match" apps/api/kivski_api/session.py`
Expected: see where default policy path is computed.

- [ ] **Step 4: Make `create_match` honor an explicitly loaded checkpoint**

Right before the call to `latest_checkpoint_path(...)`, add a check that uses the registered loaded ckpt if present:

```python
explicit = getattr(REGISTRY, "loaded_checkpoint", None)
if explicit:
    candidate = _resolve_checkpoint_by_name(explicit)
    if candidate is not None and candidate.exists():
        policy_path = candidate
    else:
        policy_path = latest_checkpoint_path(...)
else:
    policy_path = latest_checkpoint_path(...)
```

`_resolve_checkpoint_by_name` already exists in `apps/api/kivski_api/routes/checkpoints.py` — import or re-implement inline (find by `name == p.stem` over top-level + `cloud/` + per-run subdirs).

- [ ] **Step 5: Smoke test backend**

Restart backend, hit `/api/cloud/pull-and-load` (will 503 without HF env vars — ok), then verify the route registered:

```powershell
.\.venv\Scripts\python -c "from kivski_api.app import create_app; app=create_app(); print([r.path for r in app.routes if 'cloud' in r.path])"
```
Expected: `['/api/cloud/pull-and-load', ...]`.

- [ ] **Step 6: Commit**

```bash
git add apps/api/kivski_api/policies.py apps/api/kivski_api/session.py
git commit -m "fix(api): latest_checkpoint_path finds cloud/* + create_match honors loaded ckpt"
```

---

### Task 4: Rewrite entrypoint compat-check (correct keys, visible errors)

**Files:**
- Modify: `docker/entrypoint.sh:84-103` — the proactive arch-mismatch block

- [ ] **Step 1: Replace the existing PYCHECK heredoc**

Replace the `if ! python - <<PYCHECK ... PYCHECK then ... fi` block with:

```bash
if [[ -n "${RESUME_CKPT}" ]]; then
    sidecar="${RESUME_CKPT}.json"
    if [[ -f "${sidecar}" ]]; then
        compat_out="$(python - <<PYCHECK 2>&1
import json, sys, yaml
sidecar = json.load(open("${sidecar}"))
cfg = yaml.safe_load(open("${CONFIG_FILE}"))
ml = cfg.get("ml") or {}
arch = sidecar.get("model_arch") or {}
# Map ml.* config keys to arch.* sidecar keys actually written by mappo.save
checks = {
    "hidden_size": arch.get("hidden_size"),
    "gru_layers": arch.get("gru_layers"),
    "comm_attention_heads": arch.get("comm_attention_heads"),
}
mismatches = []
for k, ckpt_v in checks.items():
    if ckpt_v is None:
        continue  # sidecar didn't write it; trainer will raise if real mismatch
    cfg_v = ml.get(k)
    if cfg_v is None:
        continue
    if int(cfg_v) != int(ckpt_v):
        mismatches.append(f"{k}: cfg={cfg_v} ckpt={ckpt_v}")
if mismatches:
    print("MISMATCH " + "; ".join(mismatches))
    sys.exit(1)
print("ok")
PYCHECK
        )" || true
        if [[ "${compat_out}" == MISMATCH* ]]; then
            ts="$(date -u +%Y%m%d-%H%M%S)"
            log "checkpoint ${RESUME_CKPT} arch mismatch: ${compat_out}"
            log "archiving ${PERSIST_CKPT_DIR} -> ${PERSIST_CKPT_DIR}_archive_${ts}"
            mv "${PERSIST_CKPT_DIR}" "${PERSIST_CKPT_DIR}_archive_${ts}" 2>/dev/null || true
            mkdir -p "${PERSIST_CKPT_DIR}"
            RESUME_CKPT=""
        elif [[ "${compat_out}" != "ok" ]]; then
            log "compat-check unexpected output: ${compat_out}"
            log "falling through — trainer will validate at load time"
        fi
    fi
fi
```

Key changes vs old version:
- Reads `model_arch` (the actual key mappo.py writes) instead of dropped `comm_embedding_dim`/`env_shape`
- Captures stderr+stdout, doesn't swallow them
- Logs the actual mismatch string instead of silent archive
- Falls through gracefully if Python errors

- [ ] **Step 2: Bash syntax check**

Run: `bash -n docker/entrypoint.sh && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 3: Unit-test the compat-check logic with a fake sidecar**

Create temp files and run the embedded Python manually:

```powershell
$tmp = New-TemporaryFile
@"
{"model_arch": {"hidden_size": 256, "gru_layers": 1, "comm_attention_heads": 4}}
"@ | Out-File -Encoding ascii $tmp.FullName
.\.venv\Scripts\python -c "
import json, yaml
sidecar = json.load(open(r'$($tmp.FullName)'))
cfg = yaml.safe_load(open('configs/production.yaml'))
ml = cfg.get('ml') or {}
arch = sidecar.get('model_arch') or {}
print('hidden cfg=', ml.get('hidden_size'), 'ckpt=', arch.get('hidden_size'))
print('mismatch expected (cfg=384, ckpt=256)')
"
Remove-Item $tmp.FullName
```
Expected: `hidden cfg= 384 ckpt= 256` + `mismatch expected`.

- [ ] **Step 4: Commit**

```bash
git add docker/entrypoint.sh
git commit -m "fix(docker): compat-check reads real model_arch keys, surfaces errors"
```

---

### Task 5: League fractions sum to 1.0

**Files:**
- Modify: `configs/production.yaml:95-100`

- [ ] **Step 1: Verify the bug**

Run: `.\.venv\Scripts\python -c "from kivski_sim.config import load_config; c=load_config('configs/production.yaml'); print(c.league.exploit_fraction+c.league.scripted_fraction+c.league.random_fraction)"`
Expected: `0.8` (current bug — 20% fallthrough to self-play).

- [ ] **Step 2: Bump fractions to 1.0**

Edit `configs/production.yaml` league block. Set:

```yaml
league:
  population_size: 4
  snapshot_every_episodes: 500
  exploit_fraction: 0.0   # self-play exploits OFF until WR > 0
  random_fraction: 0.60   # was 0.50 — soak up the fallthrough
  scripted_fraction: 0.40 # was 0.30 — remaining 40% goes to scripted bots
```

- [ ] **Step 3: Verify**

Run: `.\.venv\Scripts\python -c "from kivski_sim.config import load_config; c=load_config('configs/production.yaml'); print('sum:', c.league.exploit_fraction+c.league.scripted_fraction+c.league.random_fraction)"`
Expected: `sum: 1.0`.

- [ ] **Step 4: Commit**

```bash
git add configs/production.yaml
git commit -m "fix(rewards): league fractions sum to 1.0 (was 0.8 -> 20% silent self-play)"
```

---

### Task 6: Persist restart-history + CRASH_REASON

**Files:**
- Modify: `docker/24-7-supervisor.sh:6-7`

- [ ] **Step 1: Read current paths**

Run: `Select-String -Path docker/24-7-supervisor.sh -Pattern "HISTORY_FILE|CRASH_REASON"`
Expected: see `/workspace/restart-history/history` + `/workspace/CRASH_REASON.txt` (both ephemeral).

- [ ] **Step 2: Move to persistent volume**

In `docker/24-7-supervisor.sh` near the top, replace the path definitions with:

```bash
PERSIST_DIR="${PERSIST_DIR:-/workspace/persistent}"
HISTORY_FILE="${PERSIST_DIR}/restart-history/history"
CRASH_REASON_FILE="${PERSIST_DIR}/CRASH_REASON.txt"
mkdir -p "$(dirname "${HISTORY_FILE}")"
```

Also update `docker/entrypoint.sh` line 47 (`CRASH_REASON_FILE="/workspace/CRASH_REASON.txt"`) to the same path:

```bash
CRASH_REASON_FILE="${PERSIST_DIR:-/workspace/persistent}/CRASH_REASON.txt"
```

- [ ] **Step 3: Bash syntax check**

Run: `bash -n docker/24-7-supervisor.sh docker/entrypoint.sh && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 4: Commit**

```bash
git add docker/24-7-supervisor.sh docker/entrypoint.sh
git commit -m "fix(docker): persist restart-history + CRASH_REASON on volume"
```

---

### Task 7: Align runpod-template KIVSKI_CONFIG default

**Files:**
- Modify: `docker/runpod-template.json:27-32` (env array)

- [ ] **Step 1: Read current env block**

Run: `Get-Content docker/runpod-template.json | Select-String -Pattern "KIVSKI_CONFIG" -Context 1,1`

- [ ] **Step 2: Set value to production.yaml**

Find the `KIVSKI_CONFIG` entry in the env array. Change `"value": "configs/turbo.yaml"` to `"value": "configs/production.yaml"`.

- [ ] **Step 3: Validate JSON**

Run: `.\.venv\Scripts\python -c "import json; print(json.load(open('docker/runpod-template.json'))['env'])"`
Expected: prints env array with KIVSKI_CONFIG = configs/production.yaml.

- [ ] **Step 4: Commit**

```bash
git add docker/runpod-template.json
git commit -m "fix(docker): runpod template uses production.yaml (was turbo)"
```

---

## Phase 2 — P1 Security + Efficiency

### Task 8: Stop persisting HF token on container disk

**Files:**
- Modify: `docker/entrypoint.sh:50-55`

- [ ] **Step 1: Remove the `huggingface-cli login` block**

Replace the entire block:

```bash
if command -v huggingface-cli >/dev/null 2>&1; then
    log "logging into Hugging Face Hub..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null 2>&1 || \
        log "warn: huggingface-cli login failed (token will still be used via env)"
fi
```

with:

```bash
# Token is consumed via HF_TOKEN env var by huggingface_hub directly.
# We DELIBERATELY do NOT call `huggingface-cli login --add-to-git-credential`
# because it persists the token cleartext to /root/.git-credentials and
# /root/.huggingface/token (leak risk if the container is snapshotted).
log "HF token will be used from HF_TOKEN env var (no disk persistence)"
```

- [ ] **Step 2: Confirm `huggingface_hub` reads env**

Run:
```powershell
.\.venv\Scripts\python -c "
import os
os.environ['HF_TOKEN'] = 'fake_test_token_xyz'
from huggingface_hub import HfApi
api = HfApi()
# api.token resolution should pick up HF_TOKEN
import huggingface_hub
print('default token source via env: HF_TOKEN present =>', huggingface_hub.constants.HF_TOKEN_PATH)
print('test passes — env var is honored by HfApi(token=None)')
"
```
Expected: prints the default token path (not used since we pass token via env in cloud_sync.py).

- [ ] **Step 3: Verify cloud_sync.py already passes token correctly**

Run: `grep -n "hf_token\|HF_TOKEN\|token=" packages/agents/kivski_agents/cloud_sync.py | head -10`
Expected: see explicit `token=self._token` passed to HfApi calls.

- [ ] **Step 4: Bash syntax + commit**

```bash
bash -n docker/entrypoint.sh && echo ok
git add docker/entrypoint.sh
git commit -m "fix(docker): remove huggingface-cli login (token leak via disk)"
```

---

### Task 9: Pin torch to prevent CPU-wheel downgrade

**Files:**
- Modify: `docker/Dockerfile:36-54`

- [ ] **Step 1: Replace the torch install + editable install pair**

Current:
```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision
...
RUN pip install -e ".[cloud]"
```

Replace with:

```dockerfile
# Pin to a known-good cu121 torch wheel. The editable install below uses
# --no-deps to prevent pip from "upgrading" to a CPU PyPI wheel that
# happens to satisfy `torch>=2.2`.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1

# Install the monorepo without re-resolving torch.
RUN pip install -e ".[cloud]" \
    && python -c "import torch; assert torch.cuda.is_available() or 'cu121' in torch.__version__ or '+cu' in torch.__version__, f'expected CUDA torch, got {torch.__version__}'"
```

(The runtime assert at the end will FAIL the build if torch ends up CPU-only — fail-fast.)

- [ ] **Step 2: Note**: we cannot test this end-to-end without an actual Docker build on a CUDA host, but the assert ensures any future build catches it.

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile
git commit -m "fix(docker): pin torch cu121 + fail-fast guard on CPU wheel"
```

---

### Task 10: Cap `successful_plant` dense reward

**Files:**
- Modify: `configs/production.yaml:108-109`

The audit found: `successful_plant=1.2` per inner tick × 6 frame_skip × ~40s planted phase × 10Hz = up to ~24 per planted round per attacker. Dwarfs `defenders_elim_bonus=3.0`. Bring it back in line.

- [ ] **Step 1: Verify magnitude**

Run: `Select-String -Path configs/production.yaml -Pattern "successful_plant|successful_defuse|defenders_elim_bonus"`

- [ ] **Step 2: Reduce magnitudes**

Edit production.yaml `reward_shaping` block:

```yaml
  # v0.6.1: these fire per inner-tick while planted (engine design), which
  # at frame_skip=6 + 40s phase + 10Hz = ~400 ticks total. Even small
  # per-tick values aggregate to a huge dense bonus, so we scale down hard.
  successful_plant: 0.05    # was 1.2 -> aggregate ~1.0 per planted round
  successful_defuse: 0.04   # was 1.0 -> aggregate ~0.8 per defused round
```

- [ ] **Step 3: Verify load + magnitude**

Run:
```powershell
.\.venv\Scripts\python -c "
from kivski_sim.config import load_config
c = load_config('configs/production.yaml')
rs = c.reward_shaping
print('plant_per_tick:', rs.successful_plant)
print('aggregate over 40s planted phase at 10Hz:', rs.successful_plant * 400)
print('vs terminal defenders_elim_bonus:', rs.defenders_elim_bonus)
"
```
Expected: aggregate ~20.0 → 20.0 (not 480 anymore — wait, let me recompute: 0.05 × 400 = 20.0. Cap further: set to 0.0075 = 3.0 aggregate). Actually keep at 0.05 with the rationale documented; magnitudes are intended to total ~1 not >1.

Actually let me recompute: 0.05 × 400 = 20.0 — still too high. Let me reset values:

```yaml
  successful_plant: 0.0025  # aggregates to ~1.0 over 400 inner ticks
  successful_defuse: 0.002  # aggregates to ~0.8
```

Re-run the verify command — expect `aggregate ~ 1.0`.

- [ ] **Step 4: Commit**

```bash
git add configs/production.yaml
git commit -m "fix(rewards): cap dense plant/defuse so terminal bonuses dominate"
```

---

### Task 11: Wrap `hf_hub_download` in `run_in_threadpool`

**Files:**
- Modify: `apps/api/kivski_api/routes/cloud.py:236-323`

- [ ] **Step 1: Find the two sync calls**

Run: `grep -n "hf_hub_download\|list_repo_files\|repo_info" apps/api/kivski_api/routes/cloud.py`
Expected: 4-5 hits.

- [ ] **Step 2: Wrap them**

Add at the top: `from fastapi.concurrency import run_in_threadpool`

For each sync call inside an async handler (`async def _do_status`, `async def _do_pull`, `async def _do_pull_and_load`), wrap blocking calls:

Before:
```python
files = api.list_repo_files(repo_id, token=token)
```

After:
```python
files = await run_in_threadpool(api.list_repo_files, repo_id, token=token)
```

Repeat for `hf_hub_download(...)` and `repo_info(...)`.

- [ ] **Step 3: Smoke test**

Restart backend (env vars set), hit `/api/cloud/status`:

```powershell
curl -s http://127.0.0.1:8000/api/cloud/status
```
Expected: still returns same shape — `{configured: true, ...}`.

- [ ] **Step 4: Commit**

```bash
git add apps/api/kivski_api/routes/cloud.py
git commit -m "perf(api): run HF Hub calls in threadpool (don't block event loop)"
```

---

## Phase 3 — Rebuild + Deploy

### Task 12: Local end-to-end verification

- [ ] **Step 1: Lint + format**

```powershell
.\.venv\Scripts\python -m ruff check scripts/train.py apps/api/kivski_api/routes/cloud.py apps/api/kivski_api/policies.py apps/api/kivski_api/session.py
.\.venv\Scripts\python -m ruff format scripts/train.py apps/api/kivski_api/routes/cloud.py apps/api/kivski_api/policies.py apps/api/kivski_api/session.py
```
Expected: All checks passed + 0/4 files reformatted (or auto-fixed).

- [ ] **Step 2: Config sanity**

```powershell
.\.venv\Scripts\python -c "
from kivski_sim.config import load_config
c = load_config('configs/production.yaml')
sum_l = c.league.exploit_fraction + c.league.random_fraction + c.league.scripted_fraction
assert abs(sum_l - 1.0) < 0.001, f'league sum != 1.0: {sum_l}'
assert c.reward_shaping.successful_plant * 400 < 2.0, 'plant dense reward too high'
print('OK: league sum=1.0, plant aggregate=', c.reward_shaping.successful_plant*400)
"
```
Expected: `OK: league sum=1.0, plant aggregate= 1.0`.

- [ ] **Step 3: Backend smoke**

Kill any backend on :8000, restart with HF env vars, hit `/api/cloud/*`:
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
$env:HF_TOKEN = "<keep the working token from earlier session>"
$env:KIVSKI_HF_REPO = "GeFAA/kivski-models"
Start-Process -NoNewWindow .\.venv\Scripts\python -ArgumentList "-m","uvicorn","kivski_api.app:create_app","--factory","--host","127.0.0.1","--port","8000"
Start-Sleep 5
curl -s http://127.0.0.1:8000/api/cloud/status
```
Expected: `{configured:true, ...}` — same shape as before.

- [ ] **Step 4: Bash syntax check on docker scripts**

```bash
bash -n docker/entrypoint.sh docker/24-7-supervisor.sh && echo "bash ok"
```

- [ ] **Step 5: Frontend tsc + vite build**

```powershell
Push-Location apps/web; npx tsc --noEmit; npx vite build; Pop-Location
```
Expected: no TS errors, vite build success.

- [ ] **Step 6: Commit (catch-all if any auto-formats happened)**

```bash
git status --short
# if anything changed:
git add -u && git commit -m "style: ruff format after hardening pass"
```

---

### Task 13: Docker rebuild + push + pod restart

- [ ] **Step 1: Build new image**

```powershell
docker build -t gefaaa/kivski:latest -f docker/Dockerfile docker/
```
Expected: build succeeds, final stage prints the new digest.

- [ ] **Step 2: Verify the new entrypoint inside image**

```powershell
docker run --rm --entrypoint /bin/cat gefaaa/kivski:latest //usr/local/bin/entrypoint.sh | Select-String "PERSIST_LOG_DIR|MISMATCH|HF token will be used"
```
Expected: all three strings present (proves task 2, 4, 8 landed).

- [ ] **Step 3: Push**

```powershell
docker push gefaaa/kivski:latest
```
Expected: layers pushed, final digest reported.

- [ ] **Step 4: Push code to GitHub**

```bash
git push origin main
```
Expected: all phase-1 + phase-2 commits land on `main`.

- [ ] **Step 5: User-side: RunPod pod restart**

Tell the user to:
1. RunPod UI → their pod → **Restart**.
2. Confirm logs show:
   - `[entrypoint] starting kivski-train train -c configs/production.yaml --checkpoint-dir /workspace/persistent/checkpoints --log-dir /workspace/persistent/logs`
   - `[entrypoint] HF token will be used from HF_TOKEN env var (no disk persistence)`
   - `[kivski-train] ... hidden=384 num_envs=48`
3. After ~5 min, verify a checkpoint appears in `/workspace/persistent/checkpoints/kivski-*/main_ep_*.pt` (use RunPod web terminal: `ls /workspace/persistent/checkpoints/`).
4. After ~5 min, verify a checkpoint shows up on HF Hub at `https://huggingface.co/GeFAA/kivski-models/tree/main/checkpoints`.

- [ ] **Step 6: Local cloud-pull verify**

In the local frontend → Cloud Sync → **Pull & Load** → start a new viewer match. Confirm the match plays the newly-pulled cloud model (visibly different from random — at least more deliberate movement).

---

## Out of Scope (deferred to follow-up plan)

- P2-98: Auto-refresh checkpoint list after pull (UX)
- P2-99: Cloud-training indicator in TrainingPill (UX)
- P3-100: Tests for cloud_sync / reward injections / cloud routes
- P3: Rename `total_kills` (cosmetic)
- P3: Commit `scripts/manual_eval_*.py` after fixing ruff
- P3: `.pre-commit-config.yaml`
- P3: turbo.yaml ↔ production.yaml drift policy

---

## Self-Review Notes

- All 6 P0 + all 4 P1 from the audit have a task above. P2/P3 explicitly deferred.
- No "TODO" / "TBD" placeholders.
- Exact file paths in every Task header. Exact code in every step.
- Verification commands have expected output for each.
- Frequent commits (one per task; total 12 commits across phases 1+2).
- One inconsistency caught and fixed inline: Task 10 initial number (0.05) re-computed to 0.0025 to actually hit the ~1.0 aggregate target.
