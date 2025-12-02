#!/usr/bin/env python3
"""
=============================================================================
LORA DATASET PIPELINE - Automated Training Data Preparation
=============================================================================
A "set-and-forget" tool that takes a social media username and produces
a folder of high-quality, upscaled, training-ready images.

Pipeline:
    1. Harvester  - Scrape Instagram posts & stories
    2. Butcher    - Extract frames from videos, filter blur
    3. Bouncer    - AI filter: remove images without person or person too small
    4. Judge      - Analyze quality, route for upscaling
    5. Polisher   - Upscale with Real-ESRGAN + face restoration

Run modes:
    python main.py          - CLI mode
    python main.py --gui    - Web GUI mode (http://localhost:8000)

Author: AI Pipeline Architect
=============================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION - Tweak these values
# =============================================================================

CONFIG = {
    # ==========================================================================
    # TARGET - Change this to the Instagram username you want to scrape
    # ==========================================================================
    "TARGET_USERS": ["leximariawe"],

    # ==========================================================================
    # AUTHENTICATION
    # ==========================================================================
    # Priority: 1. Browser cookies (Chrome/Firefox), 2. cookies.json file, 3. Session file
    "COOKIES_JSON_PATH": "./cookies.json",  # Fallback cookie file
    "INSTAGRAM_SESSION_USER": "",            # Optional: instaloader session username

    # ==========================================================================
    # DIRECTORIES
    # ==========================================================================
    "RAW_DOWNLOADS_DIR": "./raw_downloads",
    "EXTRACTED_FRAMES_DIR": "./extracted_frames",
    "PENDING_UPSCALE_DIR": "./pending_upscale",
    "FINAL_DATASET_DIR": "./final_dataset",

    # ==========================================================================
    # PARALLEL PROCESSING (Speed optimization)
    # ==========================================================================
    "PARALLEL_DOWNLOADS": 4,      # Concurrent post downloads
    "PARALLEL_PROCESSING": 4,     # Concurrent video/image processing

    # ==========================================================================
    # FRAME EXTRACTION (from videos/stories)
    # ==========================================================================
    "FRAME_EXTRACTION_MODE": "first_only",  # "first_only" or "interval"
    "FRAME_INTERVAL_SECONDS": 0.5,
    "BLUR_THRESHOLD": 100.0,

    # ==========================================================================
    # AI CONTENT FILTERING (Bouncer - YOLOv8)
    # ==========================================================================
    "ENABLE_PERSON_FILTER": True,     # Enable/disable person detection filter
    "MIN_PERSON_RATIO": 0.05,         # Person must be at least 5% of image area
    "DETECTION_CONFIDENCE": 0.5,      # YOLO confidence threshold
    "YOLO_MODEL": "yolov8n.pt",       # yolov8n.pt (fast) or yolov8s.pt (accurate)

    # ==========================================================================
    # QUALITY THRESHOLDS (for upscaling decisions)
    # ==========================================================================
    "MIN_RESOLUTION_NO_UPSCALE": 2048,
    "MIN_RESOLUTION_2X_UPSCALE": 1024,

    # ==========================================================================
    # UPSCALING (Real-ESRGAN)
    # ==========================================================================
    "UPSCALE_MODEL": "realesrgan-x4plus",
    "FACE_ENHANCE": True,
    "FACE_ENHANCE_MODEL": "CodeFormer",

    # ==========================================================================
    # SCRAPING SETTINGS
    # ==========================================================================
    "DOWNLOAD_POSTS": True,
    "DOWNLOAD_STORIES": True,
    "DOWNLOAD_VIDEOS": True,
    "MAX_POSTS": 0,  # 0 = Alle Posts holen (unlimited)

    # ==========================================================================
    # PROCESSING
    # ==========================================================================
    "SKIP_EXISTING": True,
    "CLEANUP_INTERMEDIATE": False,
}

# =============================================================================
# IMPORTS
# =============================================================================

from modules.harvester import Harvester
from modules.deduplicator import Deduplicator
from modules.butcher import Butcher
from modules.bouncer import Bouncer
from modules.judge import Judge
from modules.polisher import Polisher
from modules.utils import setup_logging, print_banner, print_summary

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config: dict = None, log_callback=None) -> dict:
    """
    Run the complete dataset preparation pipeline.

    Args:
        config: Optional config override
        log_callback: Optional callback for GUI logging

    Returns:
        dict with pipeline statistics
    """
    if config is None:
        config = CONFIG

    logger = setup_logging()

    # Custom log handler for GUI
    if log_callback:
        import logging
        class CallbackHandler(logging.Handler):
            def emit(self, record):
                log_callback(self.format(record))

        handler = CallbackHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger("pipeline").addHandler(handler)

    start_time = datetime.now()
    stats = {
        "downloaded_images": 0,
        "downloaded_videos": 0,
        "duplicates_removed": 0,
        "extracted_frames": 0,
        "discarded_blurry": 0,
        "filtered_no_person": 0,
        "filtered_person_small": 0,
        "upscaled_4x": 0,
        "upscaled_2x": 0,
        "no_upscale_needed": 0,
        "faces_enhanced": 0,
        "errors": 0
    }

    # Create directories
    for dir_key in ["RAW_DOWNLOADS_DIR", "EXTRACTED_FRAMES_DIR", "PENDING_UPSCALE_DIR", "FINAL_DATASET_DIR"]:
        Path(config[dir_key]).mkdir(parents=True, exist_ok=True)

    try:
        # =====================================================================
        # STAGE 1: HARVESTER
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 1: HARVESTER - Downloading content")
        logger.info("=" * 60)

        harvester = Harvester(config)
        harvest_stats = harvester.run()
        stats["downloaded_images"] = harvest_stats.get("images", 0)
        stats["downloaded_videos"] = harvest_stats.get("videos", 0)

        # =====================================================================
        # STAGE 1.5: DEDUPLICATOR - Remove duplicate images
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 1.5: DEDUPLICATOR - Removing duplicates")
        logger.info("=" * 60)

        deduplicator = Deduplicator(config)
        dedup_stats = deduplicator.run()
        stats["duplicates_removed"] = dedup_stats.get("duplicates_removed", 0)

        # =====================================================================
        # STAGE 2: BUTCHER
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 2: BUTCHER - Extracting frames from videos")
        logger.info("=" * 60)

        butcher = Butcher(config)
        butcher_stats = butcher.run()
        stats["extracted_frames"] = butcher_stats.get("extracted", 0)
        stats["discarded_blurry"] = butcher_stats.get("discarded_blurry", 0)

        # =====================================================================
        # STAGE 3: BOUNCER (AI Content Filter)
        # =====================================================================
        if config.get("ENABLE_PERSON_FILTER", True):
            logger.info("=" * 60)
            logger.info("STAGE 3: BOUNCER - AI content filtering")
            logger.info("=" * 60)

            bouncer = Bouncer(config)
            bouncer_stats = bouncer.run()
            stats["filtered_no_person"] = bouncer_stats.get("deleted_no_person", 0)
            stats["filtered_person_small"] = bouncer_stats.get("deleted_too_small", 0)
        else:
            logger.info("=" * 60)
            logger.info("STAGE 3: BOUNCER - Skipped (disabled)")
            logger.info("=" * 60)

        # =====================================================================
        # STAGE 4: JUDGE
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 4: JUDGE - Analyzing image quality")
        logger.info("=" * 60)

        judge = Judge(config)
        judge_stats = judge.run()
        stats["upscaled_4x"] = judge_stats.get("needs_4x", 0)
        stats["upscaled_2x"] = judge_stats.get("needs_2x", 0)
        stats["no_upscale_needed"] = judge_stats.get("no_upscale", 0)

        # =====================================================================
        # STAGE 5: POLISHER
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 5: POLISHER - Upscaling and face enhancement")
        logger.info("=" * 60)

        polisher = Polisher(config)
        polisher_stats = polisher.run()
        stats["faces_enhanced"] = polisher_stats.get("faces_enhanced", 0)
        stats["errors"] = polisher_stats.get("errors", 0)

        # =====================================================================
        # CLEANUP
        # =====================================================================
        if config.get("CLEANUP_INTERMEDIATE", False):
            import shutil
            shutil.rmtree(config["EXTRACTED_FRAMES_DIR"], ignore_errors=True)
            shutil.rmtree(config["PENDING_UPSCALE_DIR"], ignore_errors=True)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        stats["errors"] += 1
        raise
    finally:
        elapsed = datetime.now() - start_time
        stats["elapsed_time"] = str(elapsed)
        print_summary(stats, elapsed)

    # Final count
    final_dir = Path(config["FINAL_DATASET_DIR"])
    stats["final_count"] = len(list(final_dir.glob("*.jpg"))) + len(list(final_dir.glob("*.png")))

    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE! Final dataset: {stats['final_count']} images")
    logger.info("=" * 60)

    return stats


def main():
    """CLI entry point."""
    print_banner()

    # Check for GUI mode
    if "--gui" in sys.argv:
        from gui import run_gui
        run_gui(CONFIG)
    else:
        run_pipeline(CONFIG)

    return 0


if __name__ == "__main__":
    sys.exit(main())
