"""Quick reproduction script for the two bugs reported by the user.

Bug 1: StrictMode double-create match (api-client.ts:636 logs 2x).
Bug 2: Training 409 when Start clicked twice quickly.

Outputs to models/logs/e2e/repro_bugs.json with the captured evidence.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

from playwright.async_api import (
    Browser,
    ConsoleMessage,
    Page,
    Response,
    async_playwright,
)

FRONTEND_URL = "http://localhost:5173"
OUT = Path("models/logs/e2e/repro_bugs.json")


async def _run(headless: bool = True) -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    console_msgs: list[dict] = []
    failed_responses: list[dict] = []
    created_match_ids: list[str] = []
    match_re = re.compile(r"\[kivski\]\s*created match\s+(\S+)")

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page: Page = await ctx.new_page()

        def on_console(msg: ConsoleMessage) -> None:
            entry = {"type": msg.type, "text": msg.text}
            console_msgs.append(entry)
            m = match_re.search(msg.text)
            if m:
                created_match_ids.append(m.group(1))

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

        page.on("console", on_console)
        page.on("response", on_response)

        # ---- Stage A: load page and wait for the StrictMode dance to settle
        await page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=15_000)
        # StrictMode mount → unmount → re-mount can take a few hundred ms;
        # then the second createMatch POST needs to resolve. Wait 4s.
        await page.wait_for_timeout(4_000)

        bug1_count = len(created_match_ids)
        print(f"[repro] bug1 distinct 'created match' logs: {bug1_count}")

        # ---- Stage B: try to start training twice quickly
        start_btn = page.locator("button", has_text="Start Training").first
        bug2_evidence: dict = {"clicks_attempted": 0, "errors_visible": []}
        if await start_btn.count() > 0:
            try:
                await start_btn.click(timeout=2_000)
                bug2_evidence["clicks_attempted"] += 1
            except Exception as exc:
                bug2_evidence["click_error"] = str(exc)
            # Race the second click before status flips
            await page.wait_for_timeout(150)
            try:
                # If the button is disabled, this throws; capture it
                await start_btn.click(timeout=1_500, force=True)
                bug2_evidence["clicks_attempted"] += 1
            except Exception as exc:
                bug2_evidence["second_click_error"] = str(exc)
            await page.wait_for_timeout(2_000)

        # collect any error from the bottom controls
        error_locs = await page.locator(".text-kivski-hp-low").all()
        for loc in error_locs:
            try:
                txt = (await loc.text_content()) or ""
                if txt.strip():
                    bug2_evidence["errors_visible"].append(txt.strip())
            except Exception:
                pass

        # stop training so the system is clean
        try:
            stop_btn = page.locator("button", has_text="Stop").first
            if await stop_btn.count() > 0:
                await stop_btn.click()
                await page.wait_for_timeout(1_000)
        except Exception:
            pass

        result = {
            "started_at": time.time(),
            "bug1": {
                "created_match_logs": created_match_ids,
                "distinct_count": len(set(created_match_ids)),
                "reproduced": len(created_match_ids) > 1,
            },
            "bug2": {
                **bug2_evidence,
                "failed_responses_409": [
                    r for r in failed_responses if r["status"] == 409
                ],
                "reproduced": any(r["status"] == 409 for r in failed_responses),
            },
            "all_failed_responses": failed_responses,
            "console_msg_count": len(console_msgs),
        }
        OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print("[repro] wrote", OUT)
        print("[repro] bug1 reproduced =", result["bug1"]["reproduced"])
        print("[repro] bug2 reproduced =", result["bug2"]["reproduced"])

        await browser.close()
        return 0


if __name__ == "__main__":
    rc = asyncio.run(_run(headless="--headed" not in sys.argv))
    sys.exit(rc)
