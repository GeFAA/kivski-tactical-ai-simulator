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

# --- 4. resolve auto-resume from persistent volume --------------------------
mkdir -p "${PERSIST_CKPT_DIR}"
if [[ -z "${RESUME_CKPT}" ]]; then
    # Prefer best.pt, else newest main_ep_*.pt
    if [[ -f "${PERSIST_CKPT_DIR}/best.pt" ]]; then
        RESUME_CKPT="${PERSIST_CKPT_DIR}/best.pt"
    else
        latest="$(ls -1t "${PERSIST_CKPT_DIR}"/main_ep_*.pt 2>/dev/null | head -1 || true)"
        if [[ -n "${latest}" ]]; then RESUME_CKPT="${latest}"; fi
    fi
fi

# --- 5. exec the trainer ----------------------------------------------------
log "starting kivski-train train -c ${CONFIG_FILE}"
if [[ -n "${RESUME_CKPT}" ]]; then
    log "resuming from ${RESUME_CKPT}"
    exec kivski-train train -c "${CONFIG_FILE}" --resume "${RESUME_CKPT}"
else
    log "no checkpoint to resume — fresh run"
    exec kivski-train train -c "${CONFIG_FILE}"
fi
