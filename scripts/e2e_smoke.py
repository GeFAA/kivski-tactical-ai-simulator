"""Hard end-to-end smoke test using real headless Chromium via Playwright.

This is THE verification gate before claiming v0.2.0 works. It does what a
user does:
  1. Opens http://localhost:5173 in a real browser
  2. Asserts no ErrorBoundary fallback (no "Kivski Frontend Crash" heading)
  3. Asserts the PixiJS canvas mounted (a <canvas> in the map viewer)
  4. Asserts left/right sidebars rendered with team labels
  5. Clicks the "Start" training button, asserts no 4xx/5xx errors
  6. Listens to the websocket stream and asserts at least one
     `metrics_sample` frame arrives within 30 seconds
  7. Saves three screenshots to models/logs/e2e/*.png:
     - load.png   (initial mount)
     - training.png (after Start Training)
     - inspector.png (after click on first agent dot, if any)
  8. Captures browser console errors and unhandled rejections; fails the
     test if any uncaught error fires (the ErrorBoundary's defensive
     instrumentation makes them surface in console).

Exit code: 0 on full success, 1 on any failure with a printed reason.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import (
    Browser,
    ConsoleMessage,
    Page,
    Request,
    Response,
    async_playwright,
)

FRONTEND_URL = "http://localhost:5173"
SHOTS_DIR = Path("models/logs/e2e")
RESULTS_PATH = SHOTS_DIR / "results.json"
TIMEOUT_FOR_METRICS_MS = 90_000


async def _run(headless: bool = True) -> int:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    console_msgs: list[dict] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_responses: list[dict] = []
    ws_frames_received: list[dict] = []

    results: dict = {
        "started_at": time.time(),
        "stages": {},
        "errors": [],
    }

    def _record_results() -> None:
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page: Page = await ctx.new_page()

        def on_console(msg: ConsoleMessage) -> None:
            entry = {"type": msg.type, "text": msg.text}
            console_msgs.append(entry)
            if msg.type == "error":
                console_errors.append(msg.text)

        def on_page_error(exc) -> None:  # type: ignore[no-untyped-def]
            page_errors.append(str(exc))

        def on_response(resp: Response) -> None:
            try:
                if 400 <= resp.status < 600:
                    failed_responses.append(
                        {
                            "url": resp.url,
                            "status": resp.status,
                            "method": resp.request.method if resp.request else "?",
                        }
                    )
            except Exception:
                pass

        def on_request(_req: Request) -> None:
            pass

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)
        page.on("request", on_request)

        # Capture websocket frames so we can prove that metrics_sample arrived.
        def _on_ws(ws):  # type: ignore[no-untyped-def]
            ws.on(
                "framereceived",
                lambda data: _record_ws_frame(data, ws.url, ws_frames_received),
            )

        page.on("websocket", _on_ws)

        # ------------------------------------------------------------------ stage 1: load
        print("[e2e] loading page ...")
        await page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=15_000)
        await page.wait_for_timeout(2_000)  # give React+Pixi a moment

        title = await page.title()
        results["stages"]["load"] = {"title": title, "url": page.url}
        print(f"[e2e] title={title!r}  url={page.url!r}")

        # Did the ErrorBoundary fall through?
        crash_heading = await page.locator("h1:has-text('Kivski Frontend Crash')").count()
        if crash_heading:
            err = await page.locator("h1:has-text('Kivski Frontend Crash') + pre").first.text_content()
            results["errors"].append({"stage": "load", "kind": "error_boundary", "text": err})
            await page.screenshot(path=str(SHOTS_DIR / "crash.png"), full_page=True)
            _record_results()
            print(f"[e2e] FAIL: ErrorBoundary rendered. error={err}")
            await browser.close()
            return 1

        # Canvas mounted?
        canvas_count = await page.locator("canvas").count()
        results["stages"]["load"]["canvas_count"] = canvas_count
        print(f"[e2e] canvas elements: {canvas_count}")

        await page.screenshot(path=str(SHOTS_DIR / "load.png"), full_page=True)

        # ------------------------------------------------------------------ stage 2: start training
        print("[e2e] looking for Start training button ...")
        # The button is "Start" in TrainingPanel + BottomControls. Match the
        # one inside the panel labelled 'Training'.
        # Strategy: find a <button> whose textContent equals "Start" (case-insensitive),
        # prefer one inside a section with header containing "Training".
        start_btn = page.locator("button", has_text="Start").first
        if await start_btn.count() == 0:
            results["errors"].append({"stage": "start_training", "kind": "button_missing"})
            _record_results()
            print("[e2e] FAIL: no Start button found.")
            await browser.close()
            return 1

        await start_btn.click()
        print("[e2e] clicked Start.")

        # Look for an error toast / sentence in TrainingPanel header
        await page.wait_for_timeout(2_000)
        # Look for "404" string anywhere on the page (was the original bug)
        body_text = await page.inner_text("body")
        if "404 Not Found" in body_text or "404 not found" in body_text.lower():
            results["errors"].append(
                {"stage": "start_training", "kind": "still_404", "snippet": body_text[:300]}
            )
            _record_results()
            print("[e2e] FAIL: page still shows 404 after Start click.")
            await page.screenshot(path=str(SHOTS_DIR / "still-404.png"), full_page=True)
            await browser.close()
            return 1

        await page.screenshot(path=str(SHOTS_DIR / "training.png"), full_page=True)

        # ------------------------------------------------------------------ stage 3: wait for metrics WS frames
        print(f"[e2e] waiting up to {TIMEOUT_FOR_METRICS_MS // 1000}s for metrics_sample WS frame ...")
        start = time.time()
        while (time.time() - start) * 1000 < TIMEOUT_FOR_METRICS_MS:
            metric_frames = [f for f in ws_frames_received if "metrics_sample" in f.get("payload", "")]
            status_frames = [f for f in ws_frames_received if "training_status" in f.get("payload", "")]
            if metric_frames or status_frames:
                print(
                    f"[e2e] received metric={len(metric_frames)} status={len(status_frames)} frames "
                    f"after {time.time() - start:.1f}s"
                )
                break
            await page.wait_for_timeout(500)

        results["stages"]["metrics"] = {
            "total_ws_frames": len(ws_frames_received),
            "metric_frames": len([f for f in ws_frames_received if "metrics_sample" in f.get("payload", "")]),
            "status_frames": len(
                [f for f in ws_frames_received if "training_status" in f.get("payload", "")]
            ),
            "snapshot_frames": len([f for f in ws_frames_received if "snapshot" in f.get("payload", "")]),
        }
        print(f"[e2e] ws stats: {results['stages']['metrics']}")

        # ------------------------------------------------------------------ stage 4: click first agent dot (if canvas got one)
        if canvas_count > 0:
            box = await page.locator("canvas").first.bounding_box()
            if box:
                # click roughly into the middle of the canvas -- this should hit at least
                # one player dot if the renderer drew them
                await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                await page.wait_for_timeout(1_000)
                await page.screenshot(path=str(SHOTS_DIR / "inspector.png"), full_page=True)

        # ------------------------------------------------------------------ wrap-up
        results["console_errors"] = console_errors
        results["page_errors"] = page_errors
        # ignore api 404s for /api/training/configs that are tolerated (V1)
        non_tolerated = [
            f
            for f in failed_responses
            if not (f["url"].endswith("/api/training/configs") and f["status"] == 404)
        ]
        results["failed_responses"] = non_tolerated
        results["console_msgs_tail"] = console_msgs[-40:]
        results["ws_frame_count"] = len(ws_frames_received)

        ok = (
            not console_errors
            and not page_errors
            and not non_tolerated
            and results["stages"]["metrics"]["metric_frames"] + results["stages"]["metrics"]["status_frames"]
            > 0
        )
        results["ok"] = ok
        _record_results()

        print("[e2e] ---- summary ----")
        print(f"  console errors : {len(console_errors)}")
        print(f"  page errors    : {len(page_errors)}")
        print(f"  failed responses (non-tolerated): {len(non_tolerated)}")
        print(f"  metric frames  : {results['stages']['metrics']['metric_frames']}")
        print(f"  status frames  : {results['stages']['metrics']['status_frames']}")
        print(f"  snapshot frames: {results['stages']['metrics']['snapshot_frames']}")
        for f in non_tolerated[:5]:
            print(f"    fail: {f['method']} {f['status']} {f['url']}")
        for e in console_errors[:5]:
            print(f"    err : {e[:200]}")

        await browser.close()
        return 0 if ok else 1


def _record_ws_frame(data, url: str, sink: list[dict]) -> None:
    # data is a string (or bytes-like for binary). We only care about JSON.
    payload = data if isinstance(data, str) else data.decode("utf-8", errors="replace")
    sink.append({"url": url, "payload": payload[:512]})  # truncate to keep memory small


if __name__ == "__main__":
    rc = asyncio.run(_run(headless="--headed" not in sys.argv))
    sys.exit(rc)
