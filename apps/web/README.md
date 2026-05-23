# Kivski Tactical AI Simulator — Web Viewer

Live tactical 2D top-down match viewer for the Kivski AI vs AI tactical sim.

Stack: React 18 + TypeScript + Vite + PixiJS v8 (Canvas/WebGL) + Zustand + Tailwind.
Talks to the FastAPI backend over `/api/*` (REST) and `/ws/match` (WebSocket).

## Install

```bash
npm install
```

## Develop

```bash
npm run dev
```

Opens on `http://127.0.0.1:5173`. Vite proxies:

- `/api/*` → `http://127.0.0.1:8000`
- `/ws/*`  → `ws://127.0.0.1:8000`

When the backend is offline the viewer still renders with the built-in
`dustline` placeholder map and waits for a live WebSocket connection
(auto-reconnect with exponential backoff).

## Build

```bash
npm run build    # tsc -b && vite build
npm run preview  # serve the production build locally
```

## Scripts

| Script              | What it does                  |
|---------------------|-------------------------------|
| `npm run dev`       | Vite dev server               |
| `npm run build`     | Type-check + production build |
| `npm run preview`   | Serve `dist/` locally         |
| `npm run lint`      | ESLint                        |
| `npm run typecheck` | `tsc --noEmit`                |

## Layout

- `src/App.tsx` — 3-column shell + WS wiring
- `src/components/MatchHeader.tsx` — round/timer/score/phase header
- `src/components/LeftSidebar.tsx` — teams + player rows
- `src/components/RightSidebar.tsx` — events / inspector / comms tabs
- `src/components/MapViewer.tsx` — PixiJS top-down renderer
- `src/components/BottomControls.tsx` — playback + training controls
- `src/components/DebugToggles.tsx` — FoV / sound / comms overlays
- `src/lib/store.ts` — Zustand store (match + UI state)
- `src/lib/api-client.ts` — REST helpers + WS subscriber
- `src/lib/map-loader.ts` — map fetch + placeholder fallback
- `src/lib/types.ts` — TS types mirroring backend `types.py`

This is a skeleton — feature panels render with placeholders until
the backend emits real snapshots over the WebSocket.
