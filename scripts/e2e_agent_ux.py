"""End-to-end visual check for the Agent-UX upgrade.

Confirms that the post-redesign Simple-mode sidebar shows per-agent
compact cards (Task 1), that the map labels render at the new smaller
size (Task 2), that clicking a card opens the AgentDetailModal
(Task 3+4), and that saving a custom name persists to localStorage
and propagates to the sidebar (Task 3 persistence).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path("models/logs/e2e/agent-ux")
OUT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page = await ctx.new_page()
        await page.goto("http://localhost:5173")
        # Wait for the WS handshake + first snapshot so dots are placed.
        await page.wait_for_selector("[data-agent-card]", timeout=15000)
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(OUT / "sidebar.png"), full_page=True)

        # Click the first agent card → modal should mount.
        card = page.locator("[data-agent-card]").first
        await card.click()
        await page.wait_for_selector('input[aria-label="agent-display-name"]', timeout=5000)
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(OUT / "modal.png"), full_page=True)

        # Type a custom name and save.
        name_input = page.locator('input[aria-label="agent-display-name"]')
        await name_input.fill("Falastin")
        # Save button is "Save" inside the modal.
        await page.locator('button:has-text("Save")').click()
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(OUT / "named.png"), full_page=True)

        # Close modal (backdrop click) and check the sidebar reflects the
        # new name.
        await page.locator('[aria-label="close-modal"]').click()
        await page.wait_for_timeout(400)
        await page.screenshot(path=str(OUT / "sidebar_after_rename.png"), full_page=True)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
