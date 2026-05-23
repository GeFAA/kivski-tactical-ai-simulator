"""Diagnose: subscribe to WS, count what frames arrive while training runs.

If metrics_sample/training_status frames arrive here, the backend
broadcaster is fine and the bug is in the frontend rendering.
If not, the broadcaster isn't pushing → fix backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import urllib.request
from collections import Counter


async def main(seconds: int = 40) -> int:
    import websockets

    # Create a match like the frontend does
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/match/new",
        method="POST",
        data=b'{"map":"dustline"}',
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        match = json.loads(r.read())
    mid = match["match_id"]
    print(f"[debug] created match {mid}")

    counts: Counter[str] = Counter()
    samples: list[dict] = []

    async with websockets.connect(f"ws://127.0.0.1:8000/ws/match/{mid}") as ws:
        end = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
            except TimeoutError:
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            ftype = str(data.get("type", "?"))
            counts[ftype] += 1
            if ftype in ("metrics_sample", "training_status") and len(samples) < 6:
                samples.append(data)

    print(f"\n[debug] WS frame counts after {seconds}s:")
    for ftype, n in counts.most_common():
        print(f"  {ftype:<24} {n}")
    print("\n[debug] sample metrics/training frames:")
    for s in samples:
        print(f"  {json.dumps(s)[:200]}")

    # Cleanup
    with contextlib.suppress(Exception):
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:8000/api/match/{mid}",
                method="DELETE",
            )
        )

    return 0


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sys.exit(asyncio.run(main(s)))
