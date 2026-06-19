# Kivski 24/7 Cloud Training

Run the trainer permanently on a rented GPU. Local frontend pulls the latest model on demand. Code stays in sync via GitHub, model weights via Hugging Face Hub.

## Architektur

```
   GitHub  ───┐                                    ┌────────►  Hugging Face Hub
  (code)      │                                    │            (model weights + metrics)
              ▼                                    │                  │
   ┌──────────────────────┐   pull on restart      │                  │
   │ RunPod RTX 4090      │ ─────────► clone repo  │                  ▼
   │ docker container     │   push checkpoints ────┘            ┌──────────────────┐
   │ kivski-train --prod  │   every 100 episodes               │ Local PC         │
   └──────────────────────┘                                    │ Frontend         │
                                                               │ "Cloud Sync"     │
                                                               │ → Pull Latest    │
                                                               └──────────────────┘
```

## 1. Einmal-Setup

### Hugging Face
1. Account auf https://huggingface.co
2. New model repo erstellen → **Private** → name z.B. `dein-user/kivski-models`
3. Token erzeugen unter https://huggingface.co/settings/tokens (Type: `write`) → kopieren

### Docker image
```bash
docker build -t DEINUSER/kivski:latest -f docker/Dockerfile docker/
docker push DEINUSER/kivski:latest
```

### RunPod
1. Account auf https://runpod.io, $10–20 Guthaben aufladen
2. **Deploy → Pods → Custom Template**, Felder aus `docker/runpod-template.json` übernehmen
3. Env vars setzen:
   - `HF_TOKEN` = dein Token
   - `KIVSKI_HF_REPO` = `dein-user/kivski-models`
   - `REPO_URL` = `https://github.com/GeFAA/kivski-tactical-ai-simulator.git` (oder dein fork)
   - `REPO_REF` = `main`
4. GPU: RTX 4090 spot (~$0.34/h), Persistent volume: 50 GB
5. Deploy. Logs zeigen `Cloud sync to HF Hub enabled: dein-user/kivski-models` sobald der Trainer hochgefahren ist.

## 2. Im laufenden Betrieb

- **Cloud läuft 24/7** mit `configs/production.yaml` (WR-fix + frequent checkpoints).
- **Auto-restart** bei Crash (max 3 in 10 min, dann stop).
- **Code-Updates**: lokal `git push origin main` → pod restart triggert `git pull` → trainer übernimmt mit dem letzten Checkpoint aus dem persistent volume.

## 3. Lokal anschauen

Backend braucht die selben Env vars um den Cloud-Status zu lesen:
```bash
$env:HF_TOKEN = "dein-token"
$env:KIVSKI_HF_REPO = "dein-user/kivski-models"
pnpm dev   # oder dein bisheriger start-command
```

Im Frontend → Settings → Training tab → **☁️ Cloud Sync** Section:
- `[Refresh status]` zeigt aktuellsten Cloud-Checkpoint + Episode
- `[Pull latest]` lädt das neueste Modell nach `models/checkpoints/cloud/`
- `[Pull & Load]` lädt + aktiviert es für die nächste Viewer-Match

## 4. Kosten

| Posten | Preis |
|--------|-------|
| RTX 4090 spot | $0.34/h |
| 24/7 für einen Monat | ~$245 |
| HF Hub (private) | gratis bis ~50 GB |
| Pause beliebig möglich (stop pod) | nur Storage ~$0.10/GB/mo |

## 5. Stoppen / Pausieren

- **Kurz pausieren**: RunPod UI → Stop Pod (Volume bleibt, beim Resume macht der Trainer mit dem letzten Checkpoint weiter)
- **Komplett aufhören**: Stop + Delete pod. Volume kannst du behalten (lädst es bei Bedarf wieder mit einem neuen Pod).

## 6. Lokal (ohne Cloud) starten

Funktioniert weiter mit identischer Config:
```bash
kivski-train --config configs/production.yaml
```
Ohne `HF_TOKEN` deaktiviert sich `cloud_sync` automatisch. Im Frontend zeigt das Cloud-Sync-Panel dann "not configured".

## Troubleshooting

| Symptom | Check |
|---------|-------|
| RunPod-Logs zeigen "HF_TOKEN missing" | Env vars im Pod-Template gesetzt? |
| Frontend "configured: false" | Backend mit den selben env vars gestartet? `huggingface_hub` lokal installiert? (`pip install huggingface_hub`) |
| Trainer crashed 3× → CRASH_REASON.txt | Logs ansehen — meist OOM bei kleinerer GPU. `training.num_envs` in `production.yaml` runterdrehen. |
| Cloud-Checkpoint inkompatibel | Wenn du `obs_dim` / Hidden-Size änderst musst du auf HF Hub den alten Repo löschen oder neuen anlegen. Metadata-check im Loader verhindert silent breakage. |

Siehe `docker/README.md` für Image-Build-Details.
