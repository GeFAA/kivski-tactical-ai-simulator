"""One-shot screenshot script to verify the v2 polish viewer.

Loads the dev frontend, waits for the live match to populate, and
captures both a full-page and a canvas-only crop into
``models/logs/e2e/polish-v2/``.
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "logs" / "e2e" / "polish-v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1200})
        page = await ctx.new_page()

        errors: list[str] = []
        page.on(
            "console",
            lambda msg: errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None,
        )
        page.on("pageerror", lambda err: errors.append(f"[pageerror] {err}"))

        await page.goto("http://localhost:5173", wait_until="networkidle")
        # Let the WS handshake + multiple rounds of events fire.
        await page.wait_for_timeout(20000)

        full_path = OUT_DIR / "full.png"
        await page.screenshot(path=str(full_path), full_page=True)

        canvas = page.locator("canvas").first
        try:
            box = await canvas.bounding_box()
            if box is None:
                print("canvas bbox unavailable", file=sys.stderr)
            else:
                canvas_path = OUT_DIR / "canvas.png"
                await canvas.screenshot(path=str(canvas_path))
                # Tight crop of the top-left agent cluster.
                close_path = OUT_DIR / "closeup.png"
                clip_w = int(min(box["width"], 700))
                clip_h = int(min(box["height"], 500))
                await page.screenshot(
                    path=str(close_path),
                    clip={
                        "x": int(box["x"]),
                        "y": int(box["y"]),
                        "width": clip_w,
                        "height": clip_h,
                    },
                )
                print(f"canvas {canvas_path} ({int(box['width'])}x{int(box['height'])})")
                print(f"closeup {close_path}")
        except Exception as e:
            print(f"canvas screenshot failed: {e}", file=sys.stderr)

        print(f"full {full_path}")
        if errors:
            print("---- console errors ----")
            for e in errors:
                print(e)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
