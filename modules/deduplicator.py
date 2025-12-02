"""
Deduplicator Module
Removes duplicate images based on perceptual hashing.
"""

import logging
import hashlib
from pathlib import Path
from PIL import Image
import imagehash

from modules.utils import get_image_extensions


class Deduplicator:
    """Removes duplicate images using perceptual hashing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("pipeline.deduplicator")
        self.raw_dir = Path(config["RAW_DOWNLOADS_DIR"])
        
        # Hash difference threshold (0 = identical, higher = more different)
        # 5-10 is good for catching near-duplicates
        self.hash_threshold = config.get("DUPLICATE_THRESHOLD", 8)
    
    def _get_image_hash(self, image_path: Path) -> imagehash.ImageHash:
        """Calculate perceptual hash of an image."""
        try:
            img = Image.open(image_path)
            # Use average hash - good balance of speed and accuracy
            return imagehash.average_hash(img)
        except Exception as e:
            self.logger.debug(f"Could not hash {image_path.name}: {e}")
            return None
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content (for exact duplicates)."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def run(self) -> dict:
        """Remove duplicate images from raw downloads."""
        stats = {"total": 0, "duplicates_removed": 0, "kept": 0}
        
        if not self.raw_dir.exists():
            return stats
        
        # Collect all images
        all_images = []
        for user_dir in self.raw_dir.iterdir():
            if user_dir.is_dir():
                for f in user_dir.iterdir():
                    if f.suffix.lower() in get_image_extensions():
                        all_images.append(f)
        
        stats["total"] = len(all_images)
        
        if not all_images:
            self.logger.info("No images to deduplicate")
            return stats
        
        self.logger.info(f"Checking {len(all_images)} images for duplicates...")
        
        # Calculate hashes
        image_hashes = {}
        for img_path in all_images:
            phash = self._get_image_hash(img_path)
            if phash is not None:
                image_hashes[img_path] = phash
        
        # Find and remove duplicates
        kept_hashes = []
        kept_files = []
        duplicates = []
        
        # Sort by file size (keep larger files, they're usually higher quality)
        sorted_images = sorted(image_hashes.keys(), key=lambda p: p.stat().st_size, reverse=True)
        
        for img_path in sorted_images:
            phash = image_hashes[img_path]
            
            # Check if this is a duplicate of any kept image
            is_duplicate = False
            for kept_hash in kept_hashes:
                # Calculate hash difference
                diff = phash - kept_hash
                if diff <= self.hash_threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                duplicates.append(img_path)
            else:
                kept_hashes.append(phash)
                kept_files.append(img_path)
        
        # Delete duplicates
        for dup_path in duplicates:
            try:
                dup_path.unlink()
                stats["duplicates_removed"] += 1
                self.logger.debug(f"  Removed duplicate: {dup_path.name}")
            except Exception as e:
                self.logger.warning(f"Could not delete {dup_path.name}: {e}")
        
        stats["kept"] = len(kept_files)
        
        self.logger.info(f"\nDeduplication complete!")
        self.logger.info(f"  Total images: {stats['total']}")
        self.logger.info(f"  Duplicates removed: {stats['duplicates_removed']}")
        self.logger.info(f"  Unique images kept: {stats['kept']}")
        
        if duplicates and len(duplicates) <= 10:
            self.logger.info(f"\nRemoved duplicates:")
            for dup in duplicates:
                self.logger.info(f"  - {dup.name}")
        elif duplicates:
            self.logger.info(f"\nRemoved {len(duplicates)} duplicate files")
        
        return stats

