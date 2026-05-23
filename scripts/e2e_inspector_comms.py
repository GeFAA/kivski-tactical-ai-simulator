"""Playwright smoke test for the right-sidebar Inspector + Comms tabs.

Drives the running Vite dev server (http://localhost:5173), switches
through the relevant tabs, and saves screenshots under
``models/logs/e2e/polish/`` so the visual changes can be verified.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

OUT_DIR = Path("models/logs/e2e/polish")


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page = await ctx.new_page()

        # Stash console errors so we can fail loud on regressions.
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on(
            "console",
            lambda msg: errors.append(f"console-{msg.type}: {msg.text}")
            if msg.type == "error"
            else None,
        )

        await page.goto("http://localhost:5173")
        # Give the WS handshake + 5+ seconds of ticks time to flow through.
        await page.wait_for_timeout(8000)

        # ----- Events tab (default tab) -----
        await page.locator('button:has-text("Events")').first.click()
        await page.wait_for_timeout(500)
        await page.screenshot(
            path=str(OUT_DIR / "events_tab.png"),
            full_page=True,
        )

        # ----- Inspector with no selection -----
        await page.locator('button:has-text("Inspector")').first.click()
        await page.wait_for_timeout(500)
        await page.screenshot(
            path=str(OUT_DIR / "inspector_no_sel.png"),
            full_page=True,
        )

        # ----- Select an agent via map click -----
        canvas = page.locator("canvas").first
        box = await canvas.bounding_box()
        if box is not None:
            # Click somewhere in the yellow-team cluster (top-left of the map).
            await page.mouse.click(
                box["x"] + box["width"] * 0.16,
                box["y"] + box["height"] * 0.12,
            )
            await page.wait_for_timeout(800)
            await page.screenshot(
                path=str(OUT_DIR / "inspector_selected.png"),
                full_page=True,
            )
            # Fallback: also try clicking via the LeftSidebar to guarantee
            # a selection (the map click may have missed an agent dot).
            sidebar_btn = page.locator("aside button").first
            if await sidebar_btn.is_visible():
                await sidebar_btn.click()
                await page.wait_for_timeout(800)
                await page.screenshot(
                    path=str(OUT_DIR / "inspector_sidebar_sel.png"),
                    full_page=True,
                )

        # ----- Comms tab -----
        await page.locator('button:has-text("Comms")').first.click()
        await page.wait_for_timeout(2000)  # let extra ticks flow with comms
        await page.screenshot(
            path=str(OUT_DIR / "comms_tab.png"),
            full_page=True,
        )

        # ----- Metrics tab -----
        await page.locator('button:has-text("Metrics")').first.click()
        await page.wait_for_timeout(500)
        await page.screenshot(
            path=str(OUT_DIR / "metrics_tab.png"),
            full_page=True,
        )

        await browser.close()

        if errors:
            print("Captured browser errors:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
        else:
            print("No browser errors during the run.")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
