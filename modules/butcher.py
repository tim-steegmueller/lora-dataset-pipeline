"""
Module 2: The Butcher
Extracts frames from videos with parallel processing.
- Stories: Only first frame (they're usually still images)
- Post videos: First frame only (configurable)
Filters out blurry frames using Laplacian variance.
"""

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from modules.utils import get_video_extensions


class Butcher:
    """Extracts frames from videos with parallel processing."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("pipeline.butcher")

        self.input_dir = Path(config["RAW_DOWNLOADS_DIR"])
        self.output_dir = Path(config["EXTRACTED_FRAMES_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Settings
        self.blur_threshold = config.get("BLUR_THRESHOLD", 100.0)
        self.max_workers = config.get("PARALLEL_PROCESSING", 4)

        # Frame extraction mode
        # "first_only" = only first frame (recommended for LoRA training)
        # "interval" = extract at intervals (more frames, more variety)
        self.extraction_mode = config.get("FRAME_EXTRACTION_MODE", "first_only")
        self.frame_interval = config.get("FRAME_INTERVAL_SECONDS", 0.5)

    def _calculate_blur(self, image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        Higher value = sharper image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _extract_first_frame(self, video_path: Path, output_path: Path) -> dict:
        """Extract only the first frame from a video."""
        result = {"extracted": 0, "discarded_blurry": 0, "error": None}

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            result["error"] = f"Could not open video: {video_path.name}"
            cap.release()
            return result

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            result["error"] = f"Could not read frame: {video_path.name}"
            return result

        # Check blur
        blur_score = self._calculate_blur(frame)
        if blur_score < self.blur_threshold:
            result["discarded_blurry"] = 1
            self.logger.debug(f"  Blurry ({blur_score:.0f}): {video_path.name}")
            return result

        # Save frame as high-quality JPEG
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result["extracted"] = 1

        return result

    def _extract_frames_interval(self, video_path: Path, output_prefix: str) -> dict:
        """Extract frames at regular intervals from a video."""
        result = {"extracted": 0, "discarded_blurry": 0, "error": None}

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            result["error"] = f"Could not open video: {video_path.name}"
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            result["error"] = f"Invalid FPS: {video_path.name}"
            cap.release()
            return result

        # Calculate frame step
        frame_step = max(1, int(fps * self.frame_interval))

        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                blur_score = self._calculate_blur(frame)

                if blur_score >= self.blur_threshold:
                    output_path = self.output_dir / f"{output_prefix}_f{saved_count:04d}.jpg"
                    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    result["extracted"] += 1
                    saved_count += 1
                else:
                    result["discarded_blurry"] += 1

            frame_idx += 1

        cap.release()
        return result

    def _process_video(self, video_path: Path, username: str) -> dict:
        """Process a single video file."""
        output_prefix = f"{username}_{video_path.stem}"

        # Always extract first frame for stories and posts
        # Stories are usually still images anyway
        if self.extraction_mode == "first_only":
            output_path = self.output_dir / f"{output_prefix}.jpg"
            return self._extract_first_frame(video_path, output_path)
        else:
            # Interval mode for more frames
            return self._extract_frames_interval(video_path, output_prefix)

    def run(self) -> dict:
        """Process all videos in parallel."""
        total_stats = {"extracted": 0, "discarded_blurry": 0, "errors": 0}

        if not self.input_dir.exists():
            self.logger.warning(f"Input directory does not exist: {self.input_dir}")
            return total_stats

        # Find all videos
        video_files = []
        for user_dir in self.input_dir.iterdir():
            if user_dir.is_dir():
                username = user_dir.name
                for f in user_dir.iterdir():
                    if f.suffix.lower() in get_video_extensions():
                        video_files.append((f, username))

        if not video_files:
            self.logger.info("No videos found to process")
            return total_stats

        self.logger.info(f"Processing {len(video_files)} videos (parallel, {self.max_workers} workers)")
        self.logger.info(f"Mode: {self.extraction_mode}")

        # Process in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_video, video_path, username): video_path
                for video_path, username in video_files
            }

            for future in as_completed(futures):
                video_path = futures[future]
                result = future.result()
                completed += 1

                if result.get("error"):
                    total_stats["errors"] += 1
                    self.logger.warning(result["error"])
                else:
                    total_stats["extracted"] += result["extracted"]
                    total_stats["discarded_blurry"] += result["discarded_blurry"]

                # Progress
                if completed % 20 == 0 or completed == len(video_files):
                    self.logger.info(f"  Progress: {completed}/{len(video_files)} videos")

        self.logger.info(f"\nFrame extraction complete!")
        self.logger.info(f"  Extracted: {total_stats['extracted']} frames")
        self.logger.info(f"  Discarded (blurry): {total_stats['discarded_blurry']}")
        self.logger.info(f"  Errors: {total_stats['errors']}")

        return total_stats
