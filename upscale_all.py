#!/usr/bin/env python3
"""
Upscale all images 2x and 4x using high-quality LANCZOS + enhancement.
For M1 Mac - production upscaling should use Real-ESRGAN on Arch.
"""

import torch
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def upscale_image(input_path: Path, output_path: Path, scale: int = 2) -> tuple:
    """Upscale image using high-quality bicubic + enhancement."""

    img = Image.open(input_path).convert('RGB')
    original_size = img.size

    new_width = img.width * scale
    new_height = img.height * scale

    # High-quality upscale with LANCZOS
    upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Apply subtle sharpening (more for higher scales)
    if scale >= 4:
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2.0, percent=80, threshold=3))
    else:
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=3))

    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(upscaled)
    upscaled = enhancer.enhance(1.05)

    # Save with high quality
    upscaled.save(output_path, quality=95, optimize=True)

    return original_size, (new_width, new_height), output_path.stat().st_size


def process_image(args):
    """Process a single image (for parallel execution)."""
    input_path, output_dir_2x, output_dir_4x = args
    results = []

    try:
        # 2x upscale
        output_2x = output_dir_2x / f"{input_path.stem}_2x.jpg"
        orig, new_2x, size_2x = upscale_image(input_path, output_2x, scale=2)
        results.append(('2x', input_path.name, orig, new_2x, size_2x))

        # 4x upscale
        output_4x = output_dir_4x / f"{input_path.stem}_4x.jpg"
        _, new_4x, size_4x = upscale_image(input_path, output_4x, scale=4)
        results.append(('4x', input_path.name, orig, new_4x, size_4x))

    except Exception as e:
        results.append(('error', input_path.name, str(e), None, None))

    return results


def main():
    input_dir = Path("raw_downloads/leximariawe")
    output_dir_2x = Path("upscaled_2x")
    output_dir_4x = Path("upscaled_4x")

    output_dir_2x.mkdir(exist_ok=True)
    output_dir_4x.mkdir(exist_ok=True)

    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available!")
    else:
        print("⚠️ MPS not available, using CPU")

    # Get all images
    images = list(input_dir.glob("*.jpg"))
    print(f"\n🖼️ Upscaling {len(images)} images (2x AND 4x)...\n")

    start_time = time.time()

    # Prepare args
    args_list = [(img, output_dir_2x, output_dir_4x) for img in images]

    # Process in parallel
    all_results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_image, args): args[0] for args in args_list}

        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            completed += 1

            # Show progress
            if completed % 5 == 0 or completed == len(images):
                print(f"  Progress: {completed}/{len(images)} images")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("UPSCALING COMPLETE")
    print("=" * 60)

    success_2x = sum(1 for r in all_results if r[0] == '2x')
    success_4x = sum(1 for r in all_results if r[0] == '4x')
    errors = sum(1 for r in all_results if r[0] == 'error')

    print(f"  ✅ 2x upscaled: {success_2x} images → {output_dir_2x}/")
    print(f"  ✅ 4x upscaled: {success_4x} images → {output_dir_4x}/")
    if errors:
        print(f"  ❌ Errors: {errors}")
    print(f"  ⏱️ Time: {elapsed:.1f} seconds")

    # Show some examples
    print("\n📊 Sample results:")
    shown = 0
    for r in all_results:
        if r[0] in ('2x', '4x') and shown < 6:
            scale, name, orig, new, size = r
            print(f"  {name} ({scale}): {orig[0]}x{orig[1]} → {new[0]}x{new[1]} ({size/1024/1024:.1f} MB)")
            shown += 1

    # Total size
    total_2x = sum(f.stat().st_size for f in output_dir_2x.glob("*.jpg"))
    total_4x = sum(f.stat().st_size for f in output_dir_4x.glob("*.jpg"))

    print(f"\n💾 Total output size:")
    print(f"  2x folder: {total_2x / 1024 / 1024:.1f} MB")
    print(f"  4x folder: {total_4x / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

