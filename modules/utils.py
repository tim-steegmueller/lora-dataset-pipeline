"""
Utility functions for the LoRA Dataset Pipeline
"""

import logging
import sys
from datetime import timedelta


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██╗      ██████╗ ██████╗  █████╗     ██████╗ ██╗██████╗ ███████╗           ║
║   ██║     ██╔═══██╗██╔══██╗██╔══██╗    ██╔══██╗██║██╔══██╗██╔════╝           ║
║   ██║     ██║   ██║██████╔╝███████║    ██████╔╝██║██████╔╝█████╗             ║
║   ██║     ██║   ██║██╔══██╗██╔══██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝             ║
║   ███████╗╚██████╔╝██║  ██║██║  ██║    ██║     ██║██║     ███████╗           ║
║   ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝           ║
║                                                                              ║
║   Automated Dataset Preparation Pipeline                                     ║
║   Harvester → Butcher → Bouncer → Judge → Polisher                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_summary(stats: dict, elapsed: timedelta):
    """Print pipeline summary."""
    filtered = stats.get('filtered_no_person', 0) + stats.get('filtered_person_small', 0)

    print("\n")
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Downloaded images:    {stats.get('downloaded_images', 0)}")
    print(f"  Downloaded videos:    {stats.get('downloaded_videos', 0)}")
    print(f"  Extracted frames:     {stats.get('extracted_frames', 0)}")
    print(f"  Discarded (blurry):   {stats.get('discarded_blurry', 0)}")
    print(f"  Filtered (no person): {stats.get('filtered_no_person', 0)}")
    print(f"  Filtered (too small): {stats.get('filtered_person_small', 0)}")
    print(f"  Upscaled 4x:          {stats.get('upscaled_4x', 0)}")
    print(f"  Upscaled 2x:          {stats.get('upscaled_2x', 0)}")
    print(f"  No upscale needed:    {stats.get('no_upscale_needed', 0)}")
    print(f"  Faces enhanced:       {stats.get('faces_enhanced', 0)}")
    print(f"  Errors:               {stats.get('errors', 0)}")
    print("-" * 60)
    print(f"  Total time:           {elapsed}")
    print("=" * 60)


def get_image_extensions() -> list[str]:
    """Return list of valid image extensions."""
    return ['.jpg', '.jpeg', '.png', '.webp', '.bmp']


def get_video_extensions() -> list[str]:
    """Return list of valid video extensions."""
    return ['.mp4', '.mov', '.avi', '.mkv', '.webm']
