"""
Module 1: The Harvester
Downloads posts (images AND videos) and stories from Instagram profiles.
Uses Playwright browser automation to bypass API restrictions.
"""

import os
import json
import logging
import asyncio
import re
from pathlib import Path
from time import sleep
from datetime import datetime
import requests

from modules.utils import get_image_extensions, get_video_extensions


class Harvester:
    """Downloads content from Instagram profiles using browser automation."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("pipeline.harvester")
        self.output_dir = Path(config["RAW_DOWNLOADS_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cookies_path = Path(config.get("COOKIES_JSON_PATH", "./cookies.json"))
        # 0 means unlimited
        max_posts_config = config.get("MAX_POSTS", 100)
        self.max_posts = max_posts_config if max_posts_config > 0 else 9999

    def _load_cookies(self) -> list[dict]:
        """Load cookies from JSON file."""
        if not self.cookies_path.exists():
            return []

        try:
            with open(self.cookies_path) as f:
                cookies = json.load(f)
            self.logger.info(f"Loaded {len(cookies)} cookies")
            return cookies
        except Exception as e:
            self.logger.error(f"Failed to load cookies: {e}")
            return []

    def _download_media(self, url: str, output_path: Path) -> bool:
        """Download a media file from URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": "https://www.instagram.com/"
            }
            response = requests.get(url, timeout=60, stream=True, headers=headers)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file is valid
            if output_path.stat().st_size < 1000:
                output_path.unlink()
                return False

            return True
        except Exception as e:
            self.logger.debug(f"Download failed: {e}")
            return False

    async def _scrape_with_playwright(self, username: str) -> dict:
        """Scrape Instagram profile using Playwright browser automation."""
        stats = {"images": 0, "videos": 0, "errors": 0}

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            self.logger.error("Playwright not installed!")
            return stats

        user_dir = self.output_dir / username
        user_dir.mkdir(exist_ok=True)

        cookies = self._load_cookies()
        if not cookies:
            self.logger.error("No cookies available!")
            return stats

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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

            # Navigate to profile
            profile_url = f"https://www.instagram.com/{username}/"
            self.logger.info(f"Navigating to {profile_url}")

            try:
                await page.goto(profile_url, wait_until="load", timeout=30000)
                await page.wait_for_timeout(3000)
            except Exception as e:
                self.logger.error(f"Failed to load profile: {e}")
                await browser.close()
                return stats

            # First collect all post links by scrolling
            self.logger.info("Collecting post links...")

            post_links = set()
            scroll_count = 0
            max_scrolls = 20
            last_count = 0
            no_new_count = 0

            while len(post_links) < self.max_posts and scroll_count < max_scrolls:
                # Get all post links
                links = await page.evaluate("""
                    () => {
                        const links = new Set();
                        document.querySelectorAll('a[href*="/p/"], a[href*="/reel/"]').forEach(a => {
                            const href = a.href;
                            if (href.includes('/p/') || href.includes('/reel/')) {
                                links.add(href);
                            }
                        });
                        return Array.from(links);
                    }
                """)

                for link in links:
                    post_links.add(link)

                self.logger.info(f"  Found {len(post_links)} posts...")

                if len(post_links) == last_count:
                    no_new_count += 1
                    if no_new_count >= 3:
                        break
                else:
                    no_new_count = 0
                last_count = len(post_links)

                if len(post_links) >= self.max_posts:
                    break

                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)
                scroll_count += 1

            self.logger.info(f"Collected {len(post_links)} post links")

            # Now visit each post and download media
            downloaded = 0
            for i, post_url in enumerate(list(post_links)[:self.max_posts]):
                if downloaded >= self.max_posts:
                    break

                try:
                    await page.goto(post_url, wait_until="load", timeout=20000)
                    await page.wait_for_timeout(2000)

                    # Check for video first
                    video_url = await page.evaluate("""
                        () => {
                            const video = document.querySelector('video');
                            if (video) {
                                return video.src || video.querySelector('source')?.src || null;
                            }
                            return null;
                        }
                    """)

                    if video_url:
                        # Download video
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{username}_{timestamp}_{i:04d}.mp4"
                        output_path = user_dir / filename

                        if self._download_media(video_url, output_path):
                            stats["videos"] += 1
                            downloaded += 1
                            self.logger.info(f"  Downloaded video: {filename}")
                    else:
                        # Get highest quality image
                        img_url = await page.evaluate("""
                            () => {
                                // Try to get the main post image
                                const imgs = document.querySelectorAll('img[src*="cdninstagram"], img[src*="fbcdn"]');
                                let best = null;
                                let bestSize = 0;

                                imgs.forEach(img => {
                                    const src = img.src || '';
                                    // Skip tiny images
                                    if (src.includes('150x150') || src.includes('s150x150') ||
                                        src.includes('44x44') || src.includes('profile')) {
                                        return;
                                    }

                                    // Prefer larger images
                                    const width = img.naturalWidth || img.width || 0;
                                    if (width > bestSize) {
                                        bestSize = width;
                                        best = src;
                                    }
                                });

                                return best;
                            }
                        """)

                        if img_url:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{username}_{timestamp}_{i:04d}.jpg"
                            output_path = user_dir / filename

                            if self._download_media(img_url, output_path):
                                stats["images"] += 1
                                downloaded += 1
                                if downloaded <= 5 or downloaded % 10 == 0:
                                    self.logger.info(f"  Downloaded: {filename}")

                    # Progress
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progress: {downloaded}/{min(len(post_links), self.max_posts)} posts processed")

                    # Rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    self.logger.debug(f"Error processing post: {e}")
                    stats["errors"] += 1
                    continue

            self.logger.info(f"Successfully downloaded {stats['images']} images, {stats['videos']} videos")

            await browser.close()

        return stats

    def _download_profile(self, username: str) -> dict:
        """Download posts from a profile."""
        self.logger.info(f"Scraping @{username} with Playwright...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            stats = loop.run_until_complete(self._scrape_with_playwright(username))
        finally:
            loop.close()

        return stats

    def run(self) -> dict:
        """Run the harvester."""
        total_stats = {"images": 0, "videos": 0, "errors": 0}

        targets = self.config.get("TARGET_USERS", [])
        if not targets:
            self.logger.error("No target users configured!")
            return total_stats

        self.logger.info(f"Harvesting {len(targets)} profile(s): {', '.join(targets)}")

        for username in targets:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"  @{username}")
            self.logger.info(f"{'='*50}")

            stats = self._download_profile(username)

            for key in total_stats:
                total_stats[key] += stats.get(key, 0)

            if len(targets) > 1:
                sleep(5)

        # Count files
        actual_images = 0
        actual_videos = 0

        for user_dir in self.output_dir.iterdir():
            if user_dir.is_dir():
                for f in user_dir.iterdir():
                    if f.suffix.lower() in get_image_extensions():
                        actual_images += 1
                    elif f.suffix.lower() in get_video_extensions():
                        actual_videos += 1

        self.logger.info(f"\nHarvest complete: {actual_images} images, {actual_videos} videos")

        return {"images": actual_images, "videos": actual_videos, "errors": total_stats["errors"]}
