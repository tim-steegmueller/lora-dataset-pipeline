"""
Module 4: The Polisher
Upscales images using Real-ESRGAN with intelligent model selection.
Applies face enhancement with CodeFormer/GFPGAN when faces are detected.

Automatically selects the best settings based on source material:
- Face photos: Real-ESRGAN + CodeFormer face restoration
- General photos: Real-ESRGAN x4plus (photorealistic)
- Anime/drawings: Real-ESRGAN anime model

Optimized for AMD GPUs (ROCm) on Arch Linux.
"""

import os
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from modules.utils import get_image_extensions


@dataclass
class ImageAnalysis:
    """Analysis results for an image."""
    has_face: bool = False
    face_count: int = 0
    face_area_ratio: float = 0.0  # How much of image is faces
    is_anime: bool = False
    avg_brightness: float = 0.0
    noise_level: float = 0.0
    recommended_model: str = "realesrgan-x4plus"
    recommended_scale: int = 4


class Polisher:
    """
    Upscales images using Real-ESRGAN with smart model selection.
    Applies face restoration when faces are detected.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("pipeline.polisher")

        self.pending_dir = Path(config["PENDING_UPSCALE_DIR"])
        self.final_dir = Path(config["FINAL_DATASET_DIR"])
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Face detection
        self.face_cascade = None
        self._init_face_detector()

        # Model paths (will be set up on first run)
        self.realesrgan_path: Optional[Path] = None
        self.models_dir: Optional[Path] = None

        # Available models and their use cases
        self.models = {
            "realesrgan-x4plus": {
                "description": "Best for photorealistic images",
                "scale": 4,
                "for_faces": False
            },
            "realesrgan-x4plus-anime": {
                "description": "Best for anime/illustrations",
                "scale": 4,
                "for_faces": False
            },
            "realesrgan-x2plus": {
                "description": "2x upscale, faster, good quality",
                "scale": 2,
                "for_faces": False
            }
        }

    def _init_face_detector(self):
        """Initialize OpenCV face detector."""
        try:
            # Use Haar cascade for face detection (fast, works offline)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                self.logger.warning("Could not load face cascade classifier")
                self.face_cascade = None
        except Exception as e:
            self.logger.warning(f"Face detector init failed: {e}")
            self.face_cascade = None

    def _detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in an image."""
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def _analyze_image(self, image_path: Path) -> ImageAnalysis:
        """
        Analyze image to determine best upscaling approach.
        """
        analysis = ImageAnalysis()

        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return analysis

            height, width = img.shape[:2]
            total_pixels = width * height

            # Detect faces
            faces = self._detect_faces(img)
            analysis.face_count = len(faces)
            analysis.has_face = len(faces) > 0

            if analysis.has_face:
                # Calculate face area ratio
                face_pixels = sum(w * h for (x, y, w, h) in faces)
                analysis.face_area_ratio = face_pixels / total_pixels

            # Analyze if anime/illustration (based on color distribution)
            # Anime typically has more uniform colors and sharp edges
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.count_nonzero(edges) / total_pixels

            # Calculate color variance (anime has lower variance)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation_std = np.std(hsv[:, :, 1])

            # Heuristic: high edges + low saturation variance = anime
            analysis.is_anime = edge_ratio > 0.1 and saturation_std < 50

            # Brightness
            analysis.avg_brightness = np.mean(gray)

            # Noise estimation (Laplacian variance)
            analysis.noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Determine recommended model
            if analysis.is_anime:
                analysis.recommended_model = "realesrgan-x4plus-anime"
            else:
                analysis.recommended_model = "realesrgan-x4plus"

        except Exception as e:
            self.logger.warning(f"Image analysis failed for {image_path.name}: {e}")

        return analysis

    def _find_realesrgan(self) -> bool:
        """Find Real-ESRGAN executable."""
        # Check common locations
        possible_paths = [
            Path.home() / "ai-pipeline" / "Real-ESRGAN" / "realesrgan-ncnn-vulkan",
            Path.home() / "Real-ESRGAN" / "realesrgan-ncnn-vulkan",
            Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
            Path("/usr/bin/realesrgan-ncnn-vulkan"),
            # AUR package location
            Path("/opt/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan"),
        ]

        for path in possible_paths:
            if path.exists():
                self.realesrgan_path = path
                self.models_dir = path.parent / "models"
                self.logger.info(f"Found Real-ESRGAN at: {path}")
                return True

        # Try to find via which
        try:
            result = subprocess.run(
                ["which", "realesrgan-ncnn-vulkan"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.realesrgan_path = Path(result.stdout.strip())
                self.models_dir = self.realesrgan_path.parent / "models"
                return True
        except Exception:
            pass

        self.logger.error(
            "Real-ESRGAN not found! Install it with:\n"
            "  yay -S realesrgan-ncnn-vulkan-bin\n"
            "Or download from: https://github.com/xinntao/Real-ESRGAN/releases"
        )
        return False

    def _upscale_with_realesrgan(
        self,
        input_path: Path,
        output_path: Path,
        model: str = "realesrgan-x4plus",
        scale: int = 4
    ) -> bool:
        """
        Upscale image using Real-ESRGAN (ncnn-vulkan version for AMD GPUs).
        """
        if not self.realesrgan_path:
            if not self._find_realesrgan():
                return False

        # Build command
        cmd = [
            str(self.realesrgan_path),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", model,
            "-s", str(scale),
            "-f", "jpg",  # Output format
        ]

        # Add model path if we know it
        if self.models_dir and self.models_dir.exists():
            cmd.extend(["-m", str(self.models_dir)])

        try:
            self.logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout per image
            )

            if result.returncode != 0:
                self.logger.error(f"Real-ESRGAN failed: {result.stderr}")
                return False

            return output_path.exists()

        except subprocess.TimeoutExpired:
            self.logger.error(f"Real-ESRGAN timeout for {input_path.name}")
            return False
        except Exception as e:
            self.logger.error(f"Real-ESRGAN error: {e}")
            return False

    def _enhance_face_gfpgan(self, input_path: Path, output_path: Path) -> bool:
        """
        Enhance faces using GFPGAN/CodeFormer.
        This requires the Python version with models.
        """
        try:
            # Try to use gfpgan if installed
            from gfpgan import GFPGANer

            # Initialize GFPGAN
            restorer = GFPGANer(
                model_path='GFPGANv1.4.pth',
                upscale=1,  # Don't upscale, just restore
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )

            # Load and process
            img = cv2.imread(str(input_path))
            _, _, output = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            cv2.imwrite(str(output_path), output)
            return True

        except ImportError:
            self.logger.debug("GFPGAN not installed, skipping face enhancement")
            return False
        except Exception as e:
            self.logger.warning(f"Face enhancement failed: {e}")
            return False

    def _enhance_face_codeformer(self, input_path: Path, output_path: Path) -> bool:
        """
        Enhance faces using CodeFormer (better quality than GFPGAN).
        """
        try:
            # Check if codeformer-cli is available
            result = subprocess.run(
                ["which", "codeformer"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return False

            cmd = [
                "codeformer",
                "-i", str(input_path),
                "-o", str(output_path.parent),
                "-w", "0.7",  # Fidelity weight (0=quality, 1=fidelity)
                "--face_upsample",
                "--bg_upsampler", "realesrgan"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0

        except Exception as e:
            self.logger.debug(f"CodeFormer not available: {e}")
            return False

    def _process_image(self, image_path: Path, scale_factor: int) -> dict:
        """Process a single image with smart settings."""
        stats = {"success": False, "face_enhanced": False, "model_used": ""}

        # Analyze image
        analysis = self._analyze_image(image_path)

        # Determine actual scale (might override based on requested)
        if scale_factor == 2:
            model = "realesrgan-x2plus"
            scale = 2
        else:
            model = analysis.recommended_model
            scale = 4

        stats["model_used"] = model

        self.logger.info(
            f"  {image_path.name}: "
            f"faces={analysis.face_count}, "
            f"anime={analysis.is_anime}, "
            f"model={model}"
        )

        # Output path
        output_name = f"{image_path.stem}_upscaled{image_path.suffix}"
        output_path = self.final_dir / output_name

        # Step 1: Upscale with Real-ESRGAN
        if not self._upscale_with_realesrgan(image_path, output_path, model, scale):
            # Fallback: just copy the original
            self.logger.warning(f"Upscaling failed, copying original: {image_path.name}")
            shutil.copy2(image_path, self.final_dir / image_path.name)
            return stats

        stats["success"] = True

        # Step 2: Face enhancement if faces detected and enabled
        if analysis.has_face and self.config.get("FACE_ENHANCE", True):
            self.logger.debug(f"  Applying face enhancement...")

            # Try CodeFormer first (better), then GFPGAN
            temp_enhanced = output_path.with_stem(output_path.stem + "_face")

            enhanced = False
            if self.config.get("FACE_ENHANCE_MODEL") == "CodeFormer":
                enhanced = self._enhance_face_codeformer(output_path, temp_enhanced)

            if not enhanced:
                enhanced = self._enhance_face_gfpgan(output_path, temp_enhanced)

            if enhanced and temp_enhanced.exists():
                # Replace with enhanced version
                temp_enhanced.replace(output_path)
                stats["face_enhanced"] = True
                self.logger.debug(f"  Face enhancement applied")

        return stats

    def _install_instructions(self):
        """Print installation instructions for Arch Linux."""
        instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  INSTALLATION REQUIRED                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Real-ESRGAN (AMD GPU via Vulkan):                                          ║
║    yay -S realesrgan-ncnn-vulkan-bin                                        ║
║                                                                              ║
║  Or manual install:                                                          ║
║    1. Download from: https://github.com/xinntao/Real-ESRGAN/releases        ║
║    2. Extract to ~/Real-ESRGAN/                                             ║
║    3. Make executable: chmod +x realesrgan-ncnn-vulkan                      ║
║                                                                              ║
║  Optional - Face Enhancement:                                                ║
║    pip install gfpgan                                                        ║
║    # Download model: GFPGANv1.4.pth                                         ║
║                                                                              ║
║  Optional - CodeFormer (better faces):                                       ║
║    pip install codeformer-pip                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(instructions)

    def run(self) -> dict:
        """Process all images in pending upscale queues."""
        stats = {"upscaled": 0, "faces_enhanced": 0, "errors": 0}

        # Check if Real-ESRGAN is available
        if not self._find_realesrgan():
            self._install_instructions()
            self.logger.error("Real-ESRGAN not found. Copying originals to final dataset.")

            # Fallback: just copy files
            for scale_dir in ["4x", "2x"]:
                src_dir = self.pending_dir / scale_dir
                if src_dir.exists():
                    for f in src_dir.iterdir():
                        if f.suffix.lower() in get_image_extensions():
                            shutil.copy2(f, self.final_dir / f.name)
                            stats["upscaled"] += 1

            return stats

        # Process 4x upscale queue
        dir_4x = self.pending_dir / "4x"
        if dir_4x.exists():
            images_4x = [f for f in dir_4x.iterdir() if f.suffix.lower() in get_image_extensions()]
            self.logger.info(f"Processing {len(images_4x)} images for 4x upscale...")

            for i, img_path in enumerate(images_4x):
                try:
                    result = self._process_image(img_path, scale_factor=4)
                    if result["success"]:
                        stats["upscaled"] += 1
                    if result["face_enhanced"]:
                        stats["faces_enhanced"] += 1
                except Exception as e:
                    self.logger.error(f"Error processing {img_path.name}: {e}")
                    stats["errors"] += 1

                if (i + 1) % 10 == 0:
                    self.logger.info(f"  Progress: {i + 1}/{len(images_4x)}")

        # Process 2x upscale queue
        dir_2x = self.pending_dir / "2x"
        if dir_2x.exists():
            images_2x = [f for f in dir_2x.iterdir() if f.suffix.lower() in get_image_extensions()]
            self.logger.info(f"Processing {len(images_2x)} images for 2x upscale...")

            for i, img_path in enumerate(images_2x):
                try:
                    result = self._process_image(img_path, scale_factor=2)
                    if result["success"]:
                        stats["upscaled"] += 1
                    if result["face_enhanced"]:
                        stats["faces_enhanced"] += 1
                except Exception as e:
                    self.logger.error(f"Error processing {img_path.name}: {e}")
                    stats["errors"] += 1

                if (i + 1) % 10 == 0:
                    self.logger.info(f"  Progress: {i + 1}/{len(images_2x)}")

        self.logger.info(f"\nUpscaling complete!")
        self.logger.info(f"  Upscaled: {stats['upscaled']} images")
        self.logger.info(f"  Faces enhanced: {stats['faces_enhanced']}")
        self.logger.info(f"  Errors: {stats['errors']}")

        return stats

