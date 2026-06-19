#!/usr/bin/env bash
# Kivski trainer entrypoint.
# - Verifies required env vars
# - Fast-forwards repo to $REPO_REF if needed
# - Re-installs the package (no-op if unchanged)
# - Execs the trainer
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/kivski}"
CONFIG_FILE="${KIVSKI_CONFIG:-configs/production.yaml}"
RESUME_CKPT="${KIVSKI_RESUME_CKPT:-}"
PERSIST_CKPT_DIR="${PERSIST_CKPT_DIR:-/workspace/persistent/checkpoints}"

log() { printf '[entrypoint %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# --- 1. required env --------------------------------------------------------
missing=0
if [[ -z "${HF_TOKEN:-}" ]]; then
    log "ERROR: HF_TOKEN is not set"
    missing=1
fi
if [[ -z "${KIVSKI_HF_REPO:-}" ]]; then
    log "ERROR: KIVSKI_HF_REPO is not set (expected e.g. user/kivski-checkpoints)"
    missing=1
fi
if [[ "${missing}" -ne 0 ]]; then
    log "Refusing to start — set HF_TOKEN and KIVSKI_HF_REPO in pod env."
    exit 1
fi

# --- 2. fast-forward repo to requested ref ----------------------------------
cd "${REPO_DIR}"
if [[ -n "${REPO_REF:-}" ]]; then
    current_ref="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    log "current ref=${current_ref} requested=${REPO_REF}"
    log "fetching latest from origin..."
    git fetch --all --tags --prune
    log "checking out ${REPO_REF}"
    git checkout "${REPO_REF}"
    # Pull only if on a branch (tags/SHAs are detached).
    if git symbolic-ref -q HEAD >/dev/null; then
        git pull --ff-only origin "${REPO_REF}" || log "warn: pull failed, continuing with checked-out tree"
    fi
fi

# --- 3. re-install (fast no-op when unchanged) ------------------------------
log "ensuring package is installed (editable)..."
pip install -e . >/dev/null 2>&1 || pip install -e .

# Login to HF so the trainer's cloud sync (if enabled) can push.
if command -v huggingface-cli >/dev/null 2>&1; then
    log "logging into Hugging Face Hub..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null 2>&1 || \
        log "warn: huggingface-cli login failed (token will still be used via env)"
fi

# --- 4. archive checkpoints if previous run crashed on incompat -------------
CRASH_REASON_FILE="/workspace/CRASH_REASON.txt"
if [[ -f "${CRASH_REASON_FILE}" ]] && grep -qi "incompatible_checkpoint\|CheckpointIncompatibleError" "${CRASH_REASON_FILE}"; then
    ts="$(date -u +%Y%m%d-%H%M%S)"
    log "previous crash was incompat — archiving ${PERSIST_CKPT_DIR} -> ${PERSIST_CKPT_DIR}_archive_${ts}"
    mv "${PERSIST_CKPT_DIR}" "${PERSIST_CKPT_DIR}_archive_${ts}" 2>/dev/null || true
    rm -f "${CRASH_REASON_FILE}"
fi
mkdir -p "${PERSIST_CKPT_DIR}"

# --- 4b. resolve auto-resume from persistent volume -------------------------
if [[ -z "${RESUME_CKPT}" ]]; then
    # Prefer best.pt, else newest main_ep_*.pt
    if [[ -f "${PERSIST_CKPT_DIR}/best.pt" ]]; then
        RESUME_CKPT="${PERSIST_CKPT_DIR}/best.pt"
    else
        latest="$(ls -1t "${PERSIST_CKPT_DIR}"/main_ep_*.pt 2>/dev/null | head -1 || true)"
        if [[ -n "${latest}" ]]; then RESUME_CKPT="${latest}"; fi
    fi
fi

# --- 4c. proactively reject incompat checkpoints (avoid crash + restart) ----
# Compare arch fields in the .pt.json sidecar against current config; if
# mismatch (e.g. hidden_size bumped), archive + start fresh.
#
# Reads the keys that mappo.py actually writes (sidecar.model_arch.*) and
# captures stderr+stdout so a real Python error doesn't get swallowed and
# misclassified as "arch mismatch".
if [[ -n "${RESUME_CKPT}" ]]; then
    sidecar="${RESUME_CKPT}.json"
    if [[ -f "${sidecar}" ]]; then
        compat_out="$(python - <<PYCHECK 2>&1
import json, sys, yaml
try:
    sidecar = json.load(open("${sidecar}"))
    cfg = yaml.safe_load(open("${CONFIG_FILE}"))
    ml = cfg.get("ml") or {}
    arch = sidecar.get("model_arch") or {}
    mismatches = []
    for k in ("hidden_size", "gru_layers", "comm_attention_heads"):
        ckpt_v = arch.get(k)
        cfg_v = ml.get(k)
        if ckpt_v is None or cfg_v is None:
            continue
        if int(cfg_v) != int(ckpt_v):
            mismatches.append(f"{k}: cfg={cfg_v} ckpt={ckpt_v}")
    if mismatches:
        print("MISMATCH " + "; ".join(mismatches))
        sys.exit(0)
    print("ok")
except Exception as e:
    print(f"ERROR {type(e).__name__}: {e}")
    sys.exit(0)
PYCHECK
)" || true
        case "${compat_out}" in
            MISMATCH*)
                ts="$(date -u +%Y%m%d-%H%M%S)"
                log "checkpoint ${RESUME_CKPT} arch mismatch (${compat_out#MISMATCH })"
                log "archiving ${PERSIST_CKPT_DIR} -> ${PERSIST_CKPT_DIR}_archive_${ts}"
                mv "${PERSIST_CKPT_DIR}" "${PERSIST_CKPT_DIR}_archive_${ts}" 2>/dev/null || true
                mkdir -p "${PERSIST_CKPT_DIR}"
                RESUME_CKPT=""
                ;;
            ok)
                ;;
            ERROR*)
                log "compat-check error: ${compat_out#ERROR }"
                log "falling through — trainer will validate at load time"
                ;;
            *)
                log "compat-check unexpected output: ${compat_out}"
                log "falling through — trainer will validate at load time"
                ;;
        esac
    fi
fi

# --- 5. exec the trainer ----------------------------------------------------
# Pass persistent dirs explicitly so writes survive pod restarts/preemption.
PERSIST_LOG_DIR="${PERSIST_LOG_DIR:-/workspace/persistent/logs}"
mkdir -p "${PERSIST_LOG_DIR}"

train_args=(train -c "${CONFIG_FILE}" --checkpoint-dir "${PERSIST_CKPT_DIR}" --log-dir "${PERSIST_LOG_DIR}")
if [[ -n "${RESUME_CKPT}" ]]; then
    log "resuming from ${RESUME_CKPT}"
    train_args+=(--resume "${RESUME_CKPT}")
else
    log "no checkpoint to resume — fresh run"
fi

log "starting kivski-train ${train_args[*]}"
exec kivski-train "${train_args[@]}"
