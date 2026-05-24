"""Capture the simple-mode training pill state (training is running)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

OUT_DIR = Path("models/logs/e2e/ui-redesign")


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page = await ctx.new_page()
        await page.goto("http://localhost:5173")
        await page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
        await page.reload()
        await page.wait_for_timeout(5000)
        # Dismiss onboarding for a cleaner shot.
        try:
            close_btn = page.locator('button[aria-label="dismiss-onboarding"]').first
            if await close_btn.count() > 0:
                await close_btn.click()
                await page.wait_for_timeout(200)
        except Exception:
            pass
        await page.screenshot(path=str(OUT_DIR / "simple-with-pill.png"), full_page=True)
        await browser.close()
    print("saved", OUT_DIR / "simple-with-pill.png")


if __name__ == "__main__":
    asyncio.run(main())
