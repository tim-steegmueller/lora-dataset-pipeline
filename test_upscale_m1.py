#!/usr/bin/env python3
"""
Test upscaling on M1 Mac using PyTorch MPS backend.
Uses a simple bicubic + sharpening approach for testing.
For production, use Real-ESRGAN on the Arch machine.
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import sys

def upscale_image(input_path: Path, output_path: Path, scale: int = 2):
    """Upscale image using high-quality bicubic + enhancement."""

    # Load image
    img = Image.open(input_path).convert('RGB')
    original_size = img.size

    # Calculate new size
    new_width = img.width * scale
    new_height = img.height * scale

    # High-quality upscale with LANCZOS
    upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Apply subtle sharpening
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=3))

    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(upscaled)
    upscaled = enhancer.enhance(1.05)

    # Save with high quality
    upscaled.save(output_path, quality=95, optimize=True)

    return original_size, (new_width, new_height)


def main():
    input_dir = Path("raw_downloads/leximariawe")
    output_dir = Path("upscaled_test")
    output_dir.mkdir(exist_ok=True)

    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available!")
        device = torch.device("mps")
    else:
        print("⚠️ MPS not available, using CPU")
        device = torch.device("cpu")

    # Get first 3 images for testing
    images = list(input_dir.glob("*.jpg"))[:3]

    print(f"\n🖼️ Testing upscaling on {len(images)} images...\n")

    for img_path in images:
        output_path = output_dir / f"{img_path.stem}_2x.jpg"

        try:
            orig_size, new_size = upscale_image(img_path, output_path, scale=2)
            print(f"✅ {img_path.name}")
            print(f"   {orig_size[0]}x{orig_size[1]} → {new_size[0]}x{new_size[1]}")
            print(f"   Saved: {output_path}")
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")

    print(f"\n🎉 Done! Check {output_dir}/ for results")
    print("\n⚠️ Note: This is basic upscaling. For AI upscaling (Real-ESRGAN),")
    print("   run the full pipeline on your Arch machine with AMD GPU.")


if __name__ == "__main__":
    main()

