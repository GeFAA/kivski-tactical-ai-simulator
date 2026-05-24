"""Playwright capture of the simple/advanced redesign + settings drawer.

Drops:
    models/logs/e2e/ui-redesign/simple.png
    models/logs/e2e/ui-redesign/drawer-open.png
    models/logs/e2e/ui-redesign/advanced.png

Run via the project venv:
    .venv/Scripts/python.exe scripts/screenshot_ui_redesign.py
"""
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
        # Start with a clean slate so the default uiMode (simple) wins.
        page = await ctx.new_page()
        await page.goto("http://localhost:5173")
        await page.evaluate(
            "() => { try { localStorage.clear(); } catch (e) {} }"
        )
        await page.reload()
        await page.wait_for_timeout(4500)

        # Onboarding tooltip appears after ~800ms — dismiss it so it
        # doesn't sit on top of the gear icon in the simple screenshot.
        try:
            close_btn = page.locator(
                'button[aria-label="dismiss-onboarding"]'
            ).first
            if await close_btn.count() > 0:
                await close_btn.click()
                await page.wait_for_timeout(300)
        except Exception:
            pass

        await page.screenshot(path=str(OUT_DIR / "simple.png"), full_page=True)

        # Open settings drawer via the gear icon.
        await page.locator('button[aria-label="Settings"]').first.click()
        await page.wait_for_timeout(700)
        await page.screenshot(
            path=str(OUT_DIR / "drawer-open.png"), full_page=True
        )

        # Switch to View tab and click Advanced card.
        await page.get_by_role("button", name="View").first.click()
        await page.wait_for_timeout(300)
        await page.get_by_role("button", name="Advanced").first.click()
        await page.wait_for_timeout(400)
        await page.locator(
            'button[aria-label="close-settings"]'
        ).first.click()
        await page.wait_for_timeout(500)
        await page.screenshot(
            path=str(OUT_DIR / "advanced.png"), full_page=True
        )

        await browser.close()
    print("screenshots saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    asyncio.run(main())
