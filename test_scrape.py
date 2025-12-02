"""Quick debug script to see what Playwright sees."""

import asyncio
import json
from pathlib import Path

async def main():
    from playwright.async_api import async_playwright

    # Load cookies
    with open("cookies.json") as f:
        cookies = json.load(f)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Show browser
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )

        # Set cookies
        playwright_cookies = []
        for cookie in cookies:
            pc = {
                "name": cookie["name"],
                "value": cookie["value"],
                "domain": cookie.get("domain", ".instagram.com"),
                "path": cookie.get("path", "/"),
            }
            if cookie.get("expirationDate"):
                pc["expires"] = cookie["expirationDate"]
            playwright_cookies.append(pc)

        await context.add_cookies(playwright_cookies)

        page = await context.new_page()

        print("Navigating to Instagram...")
        await page.goto("https://www.instagram.com/leximariawe/", wait_until="load", timeout=30000)

        # Wait for page to load
        await page.wait_for_timeout(5000)

        # Take screenshot
        await page.screenshot(path="debug_screenshot.png", full_page=True)
        print("Screenshot saved to debug_screenshot.png")

        # Print page title
        title = await page.title()
        print(f"Page title: {title}")

        # Print page content (first 2000 chars)
        content = await page.content()
        print(f"Page content (first 2000 chars):\n{content[:2000]}")

        # Wait so you can see the browser
        print("\nBrowser is open. Press Ctrl+C to close...")
        await page.wait_for_timeout(30000)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

