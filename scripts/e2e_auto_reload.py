"""End-to-end verification for the per-round auto-reload feature.

Drives the full system the same way a real user would:

  1. POST /api/match/new with auto_reload_yellow=True + policy_yellow="latest".
  2. Subscribe to /ws/match/<id> and confirm the response echoed the flag.
  3. Drop a brand-new ``.pt`` file into ``models/checkpoints/`` whose mtime
     is fresher than the currently-loaded one. The watcher in the running
     session will pick it up at the next round-end.
  4. Wait for a ``policy_reload`` frame on the WebSocket and assert that
     it names the file we just created.
  5. Clean up the temp checkpoint and the match.

Exit code 0 on full success, 1 on any assertion failure with the reason
printed to stderr. Designed to be run against an already-running backend
on ``http://127.0.0.1:8000`` (the e2e harness restarts the server with
fresh code before invoking us).

Usage::

    python -m scripts.e2e_auto_reload
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

try:
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover -- script-only dependency
    from websockets.client import connect as ws_connect  # type: ignore[no-redef]

API_BASE = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"
CKPT_DIR = Path("models/checkpoints")
ROUND_END_TIMEOUT_S = 90.0


def _post_json(path: str, body: dict) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url=f"{API_BASE}{path}",
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _delete(path: str) -> int:
    req = urllib.request.Request(url=f"{API_BASE}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status
    except HTTPError as exc:
        return exc.code


def _publish_fake_checkpoint() -> Path:
    """Copy the existing ``best.pt`` to a new file with a fresher mtime.

    The CheckpointPolicy only inspects the file's existence + path for the
    hot-swap path; the actual torch.load happens lazily on the next act()
    so any byte-identical copy is safe. We deliberately do *not* run
    torch.save here because that pulls torch into the e2e harness even
    when the user already proved the trainer can produce checkpoints.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    src = CKPT_DIR / "best.pt"
    if not src.is_file():
        # Drop a tiny placeholder if there's no existing ckpt. The
        # adapter will still fail-soft to RandomPolicy on load -- the
        # discovery+swap path is what we're verifying.
        src.write_bytes(b"dummy")
        print(f"[e2e] no best.pt found; wrote {len(b'dummy')}-byte placeholder")
    name = f"e2e-fake-{int(time.time())}.pt"
    dst = CKPT_DIR / name
    shutil.copyfile(src, dst)
    # Touch the new file 5s into the future so even tests that ran two
    # operations within the same second still see it as newer.
    fresh = time.time() + 5.0
    os.utime(dst, (fresh, fresh))
    print(f"[e2e] created fake newer checkpoint: {dst}")
    return dst


async def _drive_match(fake_ckpt_name_holder: list[Path]) -> int:
    print("[e2e] creating match with auto_reload_yellow=true ...")
    body = {
        "map": "dustline",
        "policy_yellow": "latest",
        "policy_blue": "latest",
        "auto_reload_yellow": True,
        "auto_reload_blue": True,
    }
    try:
        result = _post_json("/api/match/new", body)
    except (HTTPError, URLError) as exc:
        print(f"[e2e] FAIL: could not create match: {exc}", file=sys.stderr)
        return 1
    match_id = result["match_id"]
    print(f"[e2e] match created: {match_id}, auto_reload_yellow={result.get('auto_reload_yellow')}")
    if not result.get("auto_reload_yellow"):
        print(
            f"[e2e] FAIL: backend did not echo auto_reload_yellow=true. Response: {result}",
            file=sys.stderr,
        )
        return 1

    # Pump the speed up so rounds finish fast (8x is the cap in the
    # match controller; we set 8 to give plenty of margin against the
    # 90s timeout below even on a slow CI runner).
    try:
        _post_json(f"/api/match/{match_id}/speed?multiplier=8", {})
        print("[e2e] match speed set to 8x")
    except (HTTPError, URLError) as exc:
        # Speed is a nice-to-have; tolerate failure but log it.
        print(f"[e2e] WARN: could not bump match speed: {exc}")

    try:
        async with ws_connect(f"{WS_BASE}/ws/match/{match_id}", max_size=2**24) as ws:
            print("[e2e] subscribed to WS")
            # Read the initial map_info + snapshot.
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if first.get("type") != "map_info":
                print(f"[e2e] WARN: expected map_info first, got {first.get('type')}")
            second = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if second.get("type") not in ("snapshot", "map_info"):
                print(f"[e2e] WARN: expected snapshot/map_info, got {second.get('type')}")

            # Publish a fake newer checkpoint *after* the WS is subscribed so
            # the very first round-end (which usually fires within 5-15s)
            # has something new to hot-swap to.
            fake = _publish_fake_checkpoint()
            fake_ckpt_name_holder.append(fake)
            expected_name = fake.stem
            print(f"[e2e] waiting for policy_reload (expected name={expected_name}) ...")

            t0 = time.time()
            received = None
            while time.time() - t0 < ROUND_END_TIMEOUT_S:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                except TimeoutError:
                    elapsed = time.time() - t0
                    print(f"[e2e] (no frame in 15s; total wait={elapsed:.1f}s)")
                    continue
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = frame.get("type")
                if kind == "policy_reload":
                    received = frame
                    break
            if received is None:
                print(
                    f"[e2e] FAIL: no policy_reload event arrived within {ROUND_END_TIMEOUT_S}s",
                    file=sys.stderr,
                )
                return 1
            elapsed = time.time() - t0
            data = received.get("data") or {}
            side = data.get("side")
            name = data.get("name")
            print(f"[e2e] received policy_reload event after {elapsed:.1f}s: {side} -> {name} OK")
            if name != expected_name:
                print(
                    f"[e2e] FAIL: expected name={expected_name!r}, got {name!r}",
                    file=sys.stderr,
                )
                return 1
            if side not in ("yellow", "blue"):
                print(f"[e2e] FAIL: invalid side in event: {side!r}", file=sys.stderr)
                return 1
            print("[e2e] PASS")
            return 0
    finally:
        # Clean up the match so we don't litter the registry.
        _delete(f"/api/match/{match_id}")


def main() -> int:
    fake_ckpts: list[Path] = []
    try:
        rc = asyncio.run(_drive_match(fake_ckpts))
    finally:
        # Best-effort cleanup of the fake checkpoint we wrote.
        for path in fake_ckpts:
            with __import__("contextlib").suppress(Exception):
                path.unlink()
                print(f"[e2e] cleaned up {path.name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
