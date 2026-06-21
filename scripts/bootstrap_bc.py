"""Pull a BC bootstrap checkpoint from HF Hub into the persistent volume.

Used by docker/entrypoint.sh when KIVSKI_BC_HF_FILE env var is set:
    python -m scripts.bootstrap_bc <hf_repo_path> <local_path>

The pod's persistent volume survives restarts, so this only does a real
download on the very first boot (or when the local path was deleted).
Subsequent restarts reuse the already-pulled file.

Env vars consumed:
- HF_TOKEN        (required) HF Hub write token (read also works)
- KIVSKI_HF_REPO  (required) e.g. "GeFAA/kivski-models"

Best-effort: also pulls <hf_repo_path>.json sidecar alongside.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: bootstrap_bc.py <hf_file> <local_path>", file=sys.stderr)
        return 2

    hf_file = sys.argv[1]
    local_path = Path(sys.argv[2])
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("KIVSKI_HF_REPO")
    if not token or not repo:
        print("error: HF_TOKEN and KIVSKI_HF_REPO must be set", file=sys.stderr)
        return 3

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"error: huggingface_hub not installed: {exc}", file=sys.stderr)
        return 4

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pt = hf_hub_download(repo_id=repo, filename=hf_file, token=token)
    except Exception as exc:
        print(f"error: failed to download {hf_file} from {repo}: {exc}", file=sys.stderr)
        return 5

    shutil.copy(pt, local_path)
    print(f"OK: copied {pt} -> {local_path}", flush=True)

    sidecar_remote = hf_file + ".json"
    sidecar_local = local_path.with_suffix(local_path.suffix + ".json")
    try:
        sc = hf_hub_download(repo_id=repo, filename=sidecar_remote, token=token)
        shutil.copy(sc, sidecar_local)
        print(f"OK: sidecar -> {sidecar_local}", flush=True)
    except Exception as exc:
        print(f"warn: sidecar fetch failed (ok if absent): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
