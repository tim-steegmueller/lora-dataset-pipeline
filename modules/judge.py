"""
Module 4: The Judge
Analyzes image quality and routes them for upscaling or directly to final dataset.
"""

import logging
import shutil
from pathlib import Path

from PIL import Image

from modules.utils import get_image_extensions


class Judge:
    """Analyzes images and decides upscaling requirements."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("pipeline.judge")

        self.raw_dir = Path(config["RAW_DOWNLOADS_DIR"])
        self.frames_dir = Path(config["EXTRACTED_FRAMES_DIR"])
        self.pending_dir = Path(config["PENDING_UPSCALE_DIR"])
        self.final_dir = Path(config["FINAL_DATASET_DIR"])

        # Create output directories
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Subdirs for different upscale factors
        (self.pending_dir / "4x").mkdir(exist_ok=True)
        (self.pending_dir / "2x").mkdir(exist_ok=True)

        # Thresholds
        self.min_no_upscale = config.get("MIN_RESOLUTION_NO_UPSCALE", 2048)
        self.min_2x_upscale = config.get("MIN_RESOLUTION_2X_UPSCALE", 1024)

    def _get_image_resolution(self, image_path: Path) -> tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            self.logger.warning(f"Could not read image {image_path.name}: {e}")
            return (0, 0)

    def _classify_image(self, width: int, height: int) -> str:
        """
        Classify image based on resolution.
        Returns: 'no_upscale', '2x', or '4x'
        """
        min_dim = min(width, height)

        if min_dim >= self.min_no_upscale:
            return "no_upscale"
        elif min_dim >= self.min_2x_upscale:
            return "2x"
        else:
            return "4x"

    def _collect_all_images(self) -> list[Path]:
        """Collect all images from raw downloads and extracted frames."""
        images = []

        # From raw downloads (original Instagram images)
        if self.raw_dir.exists():
            for user_dir in self.raw_dir.iterdir():
                if user_dir.is_dir():
                    for f in user_dir.iterdir():
                        if f.suffix.lower() in get_image_extensions():
                            images.append(f)

        # From extracted frames
        if self.frames_dir.exists():
            for f in self.frames_dir.iterdir():
                if f.suffix.lower() in get_image_extensions():
                    images.append(f)

        return images

    def _process_image(self, image_path: Path) -> str:
        """Process a single image and route it appropriately."""
        width, height = self._get_image_resolution(image_path)

        if width == 0 or height == 0:
            return "error"

        classification = self._classify_image(width, height)

        # Generate output filename
        output_name = image_path.name

        # Route based on classification
        if classification == "no_upscale":
            # Copy directly to final dataset
            dest = self.final_dir / output_name
            shutil.copy2(image_path, dest)
            self.logger.debug(f"  {output_name}: {width}x{height} -> final (no upscale)")

        elif classification == "2x":
            # Copy to 2x upscale queue
            dest = self.pending_dir / "2x" / output_name
            shutil.copy2(image_path, dest)
            self.logger.debug(f"  {output_name}: {width}x{height} -> 2x upscale")

        else:  # 4x
            # Copy to 4x upscale queue
            dest = self.pending_dir / "4x" / output_name
            shutil.copy2(image_path, dest)
            self.logger.debug(f"  {output_name}: {width}x{height} -> 4x upscale")

        return classification

    def run(self) -> dict:
        """Analyze all images and route them."""
        stats = {"no_upscale": 0, "needs_2x": 0, "needs_4x": 0, "errors": 0}

        images = self._collect_all_images()

        if not images:
            self.logger.warning("No images found to analyze!")
            return stats

        self.logger.info(f"Analyzing {len(images)} images...")

        for i, image_path in enumerate(images):
            try:
                result = self._process_image(image_path)

                if result == "no_upscale":
                    stats["no_upscale"] += 1
                elif result == "2x":
                    stats["needs_2x"] += 1
                elif result == "4x":
                    stats["needs_4x"] += 1
                else:
                    stats["errors"] += 1

                # Progress
                if (i + 1) % 50 == 0:
                    self.logger.info(f"  Processed {i + 1}/{len(images)} images...")

            except Exception as e:
                self.logger.error(f"Error processing {image_path.name}: {e}")
                stats["errors"] += 1
                continue

        self.logger.info(f"\nQuality analysis complete!")
        self.logger.info(f"  High-res (no upscale): {stats['no_upscale']}")
        self.logger.info(f"  Need 2x upscale: {stats['needs_2x']}")
        self.logger.info(f"  Need 4x upscale: {stats['needs_4x']}")
        self.logger.info(f"  Errors: {stats['errors']}")

        return stats

