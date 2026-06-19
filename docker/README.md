# Kivski 24/7 cloud training on RunPod

## 1. Prereqs

- Hugging Face account with a **write** access token
- Private HF model repo created (e.g. `your-user/kivski-checkpoints`)
- RunPod account with billing enabled
- Docker Hub (or other registry) account for the image

## 2. Build + push the image (one-time)

```bash
docker build \
  --build-arg REPO_URL=https://github.com/GeFAA/kivski-tactical-ai-simulator.git \
  --build-arg REPO_REF=main \
  -t YOURDOCKERUSER/kivski:latest \
  -f docker/Dockerfile docker/
docker push YOURDOCKERUSER/kivski:latest
```

Rebuild only when system deps change; code updates are pulled at container start via `REPO_REF`.

## 3. Create the RunPod pod

1. RunPod dashboard -> *Pods* -> *Deploy* -> *Custom template*.
2. Paste fields from `docker/runpod-template.json` (or import it via the API).
3. Fill the env vars:
   - `HF_TOKEN` — your HF write token
   - `KIVSKI_HF_REPO` — `your-user/kivski-checkpoints`
   - `REPO_REF` — branch/tag/SHA (default `main`)
4. Pick an RTX 4090 (spot is fine — supervisor handles restarts).
5. Deploy.

## 4. Verify

- RunPod UI -> *Logs*. Expect:
  - `[entrypoint] starting kivski-train --config configs/turbo.yaml`
  - `Cloud sync to HF Hub enabled` (once trainer initializes HF sync)
- HF repo should receive a checkpoint commit within the first checkpoint interval.

## 5. Cost estimate

RTX 4090 spot ~ **$0.34/h** -> ~ **$245/mo** for 24/7. On-demand is roughly 2x. Set RunPod budget alerts.

## 6. Stopping

Just *Stop* the pod from the RunPod UI. The persistent volume at `/workspace/persistent` survives; the next start resumes from the latest HF checkpoint.

## 7. Switching to local

```bash
kivski-train --config configs/turbo.yaml
```

Without `HF_TOKEN` / `KIVSKI_HF_REPO` set, cloud sync auto-disables and checkpoints stay local under `models/`.

## Restart-storm protection

The supervisor (`docker/24-7-supervisor.sh`) tracks restarts in `/workspace/restart-history/history`. If 3 crashes occur within 600s, it writes `/workspace/CRASH_REASON.txt` and exits 1 — RunPod will mark the pod failed instead of looping forever.
