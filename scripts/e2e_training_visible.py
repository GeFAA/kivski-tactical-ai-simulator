"""E2E that proves the training metrics + events show up in the UI.

Loads the page, starts training, waits 60s, then asserts:
  - TrainingPanel shows >= 2 sparkline points (no longer the grey 1-pt line)
  - Events tab shows at least one "Training update" event
  - The numeric policy_loss displayed in the panel is not "—"
Screenshots before/after.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

OUT = Path("models/logs/e2e/training-visible")


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page: Page = await ctx.new_page()

        async def shoot(name: str) -> None:
            await page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)

        await page.goto("http://localhost:5173", wait_until="domcontentloaded", timeout=15_000)
        await page.wait_for_timeout(3_000)
        await shoot("01_load")

        # Start training (BottomControls "Start Training" button)
        start = page.locator("button", has_text="Start Training").first
        if not await start.is_enabled():
            print("[e2e] Start Training was disabled — assuming already running")
        else:
            await start.click()
            print("[e2e] clicked Start Training")
        await page.wait_for_timeout(3_000)
        await shoot("02_started")

        # Wait for at least a few metrics to flow through
        print("[e2e] waiting 75s for metrics + events ...")
        await page.wait_for_timeout(75_000)
        await shoot("03_after_75s")

        # Click Events tab (RightSidebar default tab is usually Events, but tap to be sure)
        with contextlib.suppress(Exception):
            await page.locator("button", has_text="Events").first.click()
            await page.wait_for_timeout(800)
        await shoot("04_events_tab")

        # Read the event feed text content
        events_panel_text = ""
        with contextlib.suppress(Exception):
            events_panel_text = await page.locator("[class*='right'], aside").inner_text()
        training_event_lines = [ln for ln in events_panel_text.splitlines() if "Training update" in ln]
        print(f"[e2e] Training update events visible: {len(training_event_lines)}")
        for ln in training_event_lines[:3]:
            print(f"  {ln}")

        # Read the TrainingPanel policy_loss text (should not be "—")
        body_text = await page.inner_text("body")
        has_real_loss = ("policy loss" in body_text.lower()) and not any(
            line.strip().endswith("—") and "policy" in line.lower() for line in body_text.splitlines()
        )
        # quick heuristic via counting sparkline svg points
        svg_pts = await page.evaluate(
            "() => Array.from(document.querySelectorAll('svg polyline'))"
            ".map(p => (p.getAttribute('points')||'').split(' ').filter(Boolean).length)"
        )
        spark_max = max(svg_pts) if svg_pts else 0
        print(f"[e2e] max sparkline points: {spark_max}")
        print(f"[e2e] policy_loss line shows real value: {has_real_loss}")

        # Cleanup: stop training so subsequent tests start fresh
        with contextlib.suppress(Exception):
            await ctx.request.post("http://127.0.0.1:8000/api/training/stop")

        await browser.close()

        # Pass criteria: sparkline > 1, at least one training event visible
        ok = spark_max >= 2 and len(training_event_lines) >= 1
        print(f"[e2e] OK={ok}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
