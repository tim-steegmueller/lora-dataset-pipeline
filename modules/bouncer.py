"""
Module 3: The Bouncer (AI Content Filtering)
Filters out images where the target person is missing or too small.
Uses YOLOv8 for person detection.

Logic:
1. Run YOLO inference on image
2. Check if class 0 ("person") is detected
3. If NO person: DELETE image
4. If person detected: Check bounding box area relative to image size
5. If person too small (< threshold): DELETE image
"""

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np

from modules.utils import get_image_extensions


class Bouncer:
    """AI-powered content filter using YOLOv8 for person detection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("pipeline.bouncer")
        
        self.input_dir = Path(config["RAW_DOWNLOADS_DIR"])
        self.frames_dir = Path(config["EXTRACTED_FRAMES_DIR"])
        
        # Detection settings
        self.min_person_ratio = config.get("MIN_PERSON_RATIO", 0.05)  # Person must be at least 5% of image
        self.confidence_threshold = config.get("DETECTION_CONFIDENCE", 0.5)
        self.max_workers = config.get("PARALLEL_PROCESSING", 4)
        
        # YOLO model
        self.model = None
        self._model_loaded = False
    
    def _load_model(self) -> bool:
        """Load YOLOv8 model."""
        if self._model_loaded:
            return True
        
        try:
            from ultralytics import YOLO
            
            # Use nano model for speed, can change to yolov8s.pt for better accuracy
            model_name = self.config.get("YOLO_MODEL", "yolov8n.pt")
            self.logger.info(f"Loading YOLO model: {model_name}")
            
            self.model = YOLO(model_name)
            self._model_loaded = True
            self.logger.info("YOLO model loaded successfully")
            return True
            
        except ImportError:
            self.logger.error(
                "ultralytics not installed! Run: pip install ultralytics\n"
                "This is required for person detection filtering."
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def _analyze_image(self, image_path: Path) -> dict:
        """
        Analyze image for person detection.
        Returns dict with detection results.
        """
        result = {
            "has_person": False,
            "person_count": 0,
            "max_person_ratio": 0.0,
            "keep": False,
            "reason": ""
        }
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                result["reason"] = "Could not load image"
                return result
            
            img_height, img_width = img.shape[:2]
            img_area = img_width * img_height
            
            # Run YOLO inference
            results = self.model(img, verbose=False, conf=self.confidence_threshold)
            
            # Check for person detections (class 0 in COCO)
            person_boxes = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_area = (x2 - x1) * (y2 - y1)
                        ratio = box_area / img_area
                        person_boxes.append({
                            "box": (x1, y1, x2, y2),
                            "area": box_area,
                            "ratio": ratio,
                            "confidence": float(box.conf[0])
                        })
            
            if not person_boxes:
                result["reason"] = "No person detected"
                return result
            
            # Find the largest person detection
            result["has_person"] = True
            result["person_count"] = len(person_boxes)
            result["max_person_ratio"] = max(p["ratio"] for p in person_boxes)
            
            # Check if person is large enough
            if result["max_person_ratio"] >= self.min_person_ratio:
                result["keep"] = True
                result["reason"] = f"Person detected ({result['max_person_ratio']*100:.1f}% of image)"
            else:
                result["reason"] = f"Person too small ({result['max_person_ratio']*100:.1f}% < {self.min_person_ratio*100:.1f}%)"
            
            return result
            
        except Exception as e:
            result["reason"] = f"Analysis error: {str(e)}"
            return result
    
    def _process_image(self, image_path: Path) -> dict:
        """Process a single image and decide keep/delete."""
        analysis = self._analyze_image(image_path)
        
        if not analysis["keep"]:
            # Delete the image
            try:
                image_path.unlink()
                return {
                    "deleted": True,
                    "reason": analysis["reason"],
                    "path": str(image_path)
                }
            except Exception as e:
                return {
                    "deleted": False,
                    "reason": f"Delete failed: {e}",
                    "path": str(image_path)
                }
        else:
            return {
                "deleted": False,
                "reason": analysis["reason"],
                "path": str(image_path),
                "person_ratio": analysis["max_person_ratio"]
            }
    
    def _collect_images(self) -> list[Path]:
        """Collect all images from raw downloads and extracted frames."""
        images = []
        
        # From raw downloads
        if self.input_dir.exists():
            for user_dir in self.input_dir.iterdir():
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
    
    def run(self) -> dict:
        """Run the bouncer on all images."""
        stats = {
            "total": 0,
            "kept": 0,
            "deleted_no_person": 0,
            "deleted_too_small": 0,
            "errors": 0
        }
        
        # Load model
        if not self._load_model():
            self.logger.error("Cannot run bouncer without YOLO model")
            return stats
        
        # Collect images
        images = self._collect_images()
        stats["total"] = len(images)
        
        if not images:
            self.logger.info("No images to filter")
            return stats
        
        self.logger.info(f"Filtering {len(images)} images for person content...")
        self.logger.info(f"Min person ratio: {self.min_person_ratio*100:.1f}% of image area")
        
        # Process images (sequential for YOLO, GPU doesn't like parallel inference)
        # But we can batch if needed
        deleted_paths = []
        kept_paths = []
        
        for i, image_path in enumerate(images):
            result = self._process_image(image_path)
            
            if result["deleted"]:
                if "No person" in result["reason"]:
                    stats["deleted_no_person"] += 1
                elif "too small" in result["reason"]:
                    stats["deleted_too_small"] += 1
                deleted_paths.append((image_path.name, result["reason"]))
            else:
                stats["kept"] += 1
                kept_paths.append(image_path.name)
            
            # Progress
            if (i + 1) % 20 == 0 or (i + 1) == len(images):
                self.logger.info(
                    f"  Progress: {i+1}/{len(images)} | "
                    f"Kept: {stats['kept']} | "
                    f"Deleted: {stats['deleted_no_person'] + stats['deleted_too_small']}"
                )
        
        # Summary
        self.logger.info(f"\nBouncer filtering complete!")
        self.logger.info(f"  Total images: {stats['total']}")
        self.logger.info(f"  Kept: {stats['kept']}")
        self.logger.info(f"  Deleted (no person): {stats['deleted_no_person']}")
        self.logger.info(f"  Deleted (person too small): {stats['deleted_too_small']}")
        
        # Log some deleted files for reference
        if deleted_paths:
            self.logger.info(f"\nDeleted files (sample):")
            for name, reason in deleted_paths[:5]:
                self.logger.info(f"  - {name}: {reason}")
            if len(deleted_paths) > 5:
                self.logger.info(f"  ... and {len(deleted_paths) - 5} more")
        
        return stats

