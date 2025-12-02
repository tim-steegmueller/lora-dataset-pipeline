# 🎨 LoRA Dataset Pipeline

Automated Instagram scraping and dataset preparation pipeline for training LoRA models.

## Features

- **🌾 Harvester** - Scrapes Instagram posts & stories using Playwright browser automation
- **🔪 Butcher** - Extracts frames from videos with blur detection (Laplacian Variance)
- **🔍 Deduplicator** - Removes duplicate images using perceptual hashing
- **🚪 Bouncer** - AI content filtering with YOLOv8 (removes images without persons)
- **⚖️ Judge** - Analyzes image quality and routes for upscaling
- **✨ Polisher** - Upscales with Real-ESRGAN + face restoration (CodeFormer/GFPGAN)

## Requirements

- Python 3.10+
- Playwright (for Instagram scraping)
- Real-ESRGAN (for upscaling - install on target machine)

## Installation

```bash
# Clone the repo
git clone https://github.com/tim-steegmueller/lora-dataset-pipeline.git
cd lora-dataset-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Configuration

Edit `main.py` to configure:

```python
CONFIG = {
    "TARGET_USERS": ["instagram_username"],
    "COOKIES_JSON_PATH": "./cookies.json",
    # ... more options
}
```

### Instagram Authentication

Create a `cookies.json` file with your Instagram session cookies (export from browser).

## Usage

### CLI Mode
```bash
python main.py
```

### Web GUI Mode
```bash
python main.py --gui
# Open http://localhost:8000
```

## Pipeline Flow

```
Instagram Profile
       ↓
   [Harvester] → Downloads all posts/stories
       ↓
  [Deduplicator] → Removes duplicate images
       ↓
   [Butcher] → Extracts frames from videos, filters blur
       ↓
   [Bouncer] → AI filtering (YOLOv8 person detection)
       ↓
    [Judge] → Quality analysis, routing
       ↓
   [Polisher] → Real-ESRGAN upscaling + face restoration
       ↓
  final_dataset/
```

## Target Platform

Optimized for:
- **AMD Ryzen 9 7900X**
- **AMD Radeon RX 7800 XT 16GB** (ROCm/Vulkan)
- **Arch Linux**

Real-ESRGAN installation on Arch:
```bash
yay -S realesrgan-ncnn-vulkan-bin
```

## License

MIT

