"""Headless Playwright proof that the FoV overlay is wall-clipped.

Spins a fresh match on the running backend (so we know we have live
agents to select), opens the viewer, toggles ``Show FoV`` on, picks
the first agent dot near the centre of the canvas, and saves a
screenshot to ``models/logs/e2e/fov-clipped.png`` for visual review.

Returns non-zero exit code if the page crashed or no canvas mounted.
The "looks correct" part is human-checked by inspecting the PNG.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

from playwright.async_api import Page, async_playwright

FRONTEND_URL = "http://localhost:5173"
BACKEND_URL = "http://localhost:8000"
SHOT_DIR = Path("models/logs/e2e")
SHOT_PATH = SHOT_DIR / "fov-clipped.png"


def _ensure_match() -> str | None:
    """POST /api/match/new and return the match id (best-effort)."""

    body = json.dumps(
        {
            "map": "dustline",
            "policy_yellow": "random",
            "policy_blue": "random",
            "autostart": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}/api/match/new",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("match_id")
    except urllib.error.HTTPError as exc:
        print(f"[fov-e2e] match/new HTTP {exc.code}: {exc.read()[:200]}")
        return None
    except Exception as exc:
        print(f"[fov-e2e] match/new failed: {exc}")
        return None


async def _run(headless: bool = True) -> int:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)

    match_id = _ensure_match()
    print(f"[fov-e2e] match_id={match_id!r}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page: Page = await ctx.new_page()

        page_errors: list[str] = []
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "console",
            lambda msg: (
                page_errors.append(f"console.error: {msg.text}")
                if msg.type == "error"
                else None
            ),
        )

        await page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=15_000)
        await page.wait_for_timeout(3_000)  # give Pixi + initial snapshot a moment

        crash_count = await page.locator("h1:has-text('Kivski Frontend Crash')").count()
        if crash_count:
            await page.screenshot(path=str(SHOT_DIR / "fov-crash.png"), full_page=True)
            print("[fov-e2e] FAIL: ErrorBoundary rendered.")
            await browser.close()
            return 1

        canvas_count = await page.locator("canvas").count()
        print(f"[fov-e2e] canvas count={canvas_count}")
        if canvas_count == 0:
            print("[fov-e2e] FAIL: no canvas mounted.")
            await browser.close()
            return 1

        # Toggle Show FoV. The DebugToggles renders <label>Show FoV<input/></label>.
        fov_toggle = page.locator("label:has-text('Show FoV')").first
        if await fov_toggle.count() == 0:
            print("[fov-e2e] FAIL: Show FoV toggle not found.")
            await page.screenshot(path=str(SHOT_DIR / "fov-no-toggle.png"), full_page=True)
            await browser.close()
            return 1
        await fov_toggle.click()
        print("[fov-e2e] clicked Show FoV.")

        # Wait for snapshots to flow.
        await page.wait_for_timeout(1_500)

        # Select the first agent via the LeftSidebar PlayerRow button so we
        # know a cone will be drawn. The sidebar rows are <button> elements
        # whose visible text starts with the agent name (e.g. "agent_0").
        # We pick the *first* sidebar button to get a deterministic target.
        sidebar_buttons = page.locator(
            "section button:has(.h-1\\.5)"
        )  # PlayerRow has an inner hp bar div with h-1.5
        if await sidebar_buttons.count() == 0:
            # Fall back: any button mentioning kills/deaths format "0/0/0".
            sidebar_buttons = page.locator("button:has-text('0/0/0')")

        btn_count = await sidebar_buttons.count()
        print(f"[fov-e2e] sidebar player buttons={btn_count}")
        if btn_count > 0:
            await sidebar_buttons.first.click()
            print("[fov-e2e] clicked first sidebar player.")
        else:
            # Last-resort: click the centre of the canvas a few times.
            canvas_fallback = page.locator("canvas").first
            cbox = await canvas_fallback.bounding_box()
            if cbox:
                for dx in (0, -40, 40, -80, 80):
                    await page.mouse.click(
                        cbox["x"] + cbox["width"] / 2 + dx,
                        cbox["y"] + cbox["height"] / 2,
                    )
                    await page.wait_for_timeout(300)

        await page.wait_for_timeout(2_000)

        canvas = page.locator("canvas").first
        await page.screenshot(path=str(SHOT_PATH), full_page=True)
        print(f"[fov-e2e] saved screenshot -> {SHOT_PATH}")

        # Tighter crop of the canvas itself so reviewers see the cone clearly.
        canvas_shot = SHOT_DIR / "fov-clipped-canvas.png"
        try:
            await canvas.screenshot(path=str(canvas_shot))
            print(f"[fov-e2e] saved canvas crop -> {canvas_shot}")
        except Exception as exc:
            print(f"[fov-e2e] canvas crop failed (non-fatal): {exc}")

        print(f"[fov-e2e] page_errors={len(page_errors)}")
        for err in page_errors[:5]:
            print(f"    {err[:300]}")

        await browser.close()
        return 0 if not page_errors else 1


if __name__ == "__main__":
    rc = asyncio.run(_run(headless="--headed" not in sys.argv))
    sys.exit(rc)
