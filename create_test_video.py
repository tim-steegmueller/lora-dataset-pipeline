#!/usr/bin/env python3
"""Create a test video from images to test frame extraction."""

from moviepy import ImageSequenceClip, ImageClip, concatenate_videoclips
from PIL import Image
from pathlib import Path
import tempfile
import os

# Get some images
images_dir = Path("raw_downloads/leximariawe")
images = sorted(images_dir.glob("*.jpg"))[:5]

if not images:
    print("No images found!")
    exit(1)

print(f"Creating test video from {len(images)} images...")

# Resize all images to same size
target_size = (1080, 1920)  # Instagram story size
temp_dir = tempfile.mkdtemp()
resized_images = []

for i, img_path in enumerate(images):
    img = Image.open(img_path)
    # Resize maintaining aspect ratio, then crop/pad to target
    img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)

    # Create new image with target size (black background)
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    # Paste resized image centered
    x = (target_size[0] - img.width) // 2
    y = (target_size[1] - img.height) // 2
    new_img.paste(img, (x, y))

    temp_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
    new_img.save(temp_path, quality=95)
    resized_images.append(temp_path)

# Create video
clip = ImageSequenceClip(resized_images, durations=[1.0] * len(resized_images))

output_path = images_dir / "test_story_video.mp4"
clip.write_videofile(str(output_path), fps=24, codec='libx264', audio=False, logger=None)

# Cleanup
for f in resized_images:
    os.remove(f)
os.rmdir(temp_dir)

print(f"✅ Created test video: {output_path}")
print(f"   Duration: {len(images)} seconds")

