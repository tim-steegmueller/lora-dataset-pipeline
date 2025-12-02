# 🎨 LoRA Dataset Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-Browser_Automation-green?style=for-the-badge&logo=playwright&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-AI_Filtering-red?style=for-the-badge&logo=yolo&logoColor=white)
![Real-ESRGAN](https://img.shields.io/badge/Real--ESRGAN-Upscaling-purple?style=for-the-badge)

**Automated Instagram scraping and dataset preparation pipeline for training LoRA models.**

*From Instagram profile to training-ready dataset in one command.*

</div>

---

## 🚀 Features

| Module | Name | Description |
|--------|------|-------------|
| 🌾 | **Harvester** | Scrapes Instagram posts & stories using Playwright browser automation |
| 🔍 | **Deduplicator** | Removes duplicate images using perceptual hashing (imagehash) |
| 🔪 | **Butcher** | Extracts frames from videos with Laplacian blur detection |
| 🚪 | **Bouncer** | AI content filtering with YOLOv8 - removes images without persons |
| ⚖️ | **Judge** | Analyzes image quality and routes for appropriate upscaling |
| ✨ | **Polisher** | Upscales with Real-ESRGAN + optional face restoration (CodeFormer/GFPGAN) |

## 📊 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Instagram Profile                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🌾 HARVESTER                                                    │
│  • Playwright browser automation (bypasses API restrictions)    │
│  • Downloads posts, reels, stories                              │
│  • Parallel downloads for speed                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔍 DEDUPLICATOR                                                 │
│  • Perceptual hashing (pHash)                                   │
│  • Removes near-identical images                                │
│  • Keeps highest quality version                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔪 BUTCHER                                                      │
│  • Extracts frames from videos/stories                          │
│  • Mode: first_only (recommended) or interval                   │
│  • Laplacian variance blur detection                            │
│  • Auto-deletes blurry frames                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🚪 BOUNCER (AI Filtering)                                       │
│  • YOLOv8 person detection                                      │
│  • Removes landscape/food/travel shots without people           │
│  • Configurable minimum person size (% of image)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ⚖️ JUDGE                                                        │
│  • Resolution analysis                                          │
│  • Routes: 4x upscale / 2x upscale / ready                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ✨ POLISHER                                                     │
│  • Real-ESRGAN upscaling (AMD GPU via Vulkan)                  │
│  • Optional: CodeFormer/GFPGAN face restoration                │
│  • High-quality JPEG output                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    📁 final_dataset/                             │
│              Training-ready high-res images                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Chrome/Firefox (for cookie extraction)
- Real-ESRGAN (for upscaling - see below)

### Quick Start

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

### Real-ESRGAN Installation

<details>
<summary><b>🐧 Arch Linux (AMD GPU - Recommended)</b></summary>

```bash
# Install via AUR (Vulkan backend for AMD)
yay -S realesrgan-ncnn-vulkan-bin

# Verify installation
realesrgan-ncnn-vulkan --help
```

</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
# Download pre-built binary
curl -L -o realesrgan-macos.zip \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"

unzip realesrgan-macos.zip -d realesrgan-bin
chmod +x realesrgan-bin/realesrgan-ncnn-vulkan
```

> ⚠️ Note: Vulkan may not work on Apple Silicon. Use the Python fallback (LANCZOS + sharpening) or run on Linux.

</details>

<details>
<summary><b>🪟 Windows</b></summary>

```bash
# Download from GitHub Releases
# https://github.com/xinntao/Real-ESRGAN/releases

# Extract and add to PATH
```

</details>

### Optional: Face Enhancement

```bash
# GFPGAN
pip install gfpgan

# CodeFormer (better quality)
pip install codeformer-pip
```

## ⚙️ Configuration

Edit `main.py` to customize the pipeline:

```python
CONFIG = {
    # Target Instagram profile(s)
    "TARGET_USERS": ["username1", "username2"],
    
    # Authentication
    "COOKIES_JSON_PATH": "./cookies.json",
    
    # Directories
    "RAW_DOWNLOADS_DIR": "./raw_downloads",
    "FINAL_DATASET_DIR": "./final_dataset",
    
    # Performance
    "PARALLEL_DOWNLOADS": 4,
    "PARALLEL_PROCESSING": 4,
    
    # Frame extraction from videos
    "FRAME_EXTRACTION_MODE": "first_only",  # or "interval"
    "FRAME_INTERVAL_SECONDS": 0.5,
    "BLUR_THRESHOLD": 100.0,
    
    # AI Content Filtering (Bouncer)
    "ENABLE_PERSON_FILTER": True,
    "MIN_PERSON_RATIO": 0.05,  # Person must be 5% of image
    "YOLO_MODEL": "yolov8n.pt",
    
    # Quality thresholds
    "MIN_RESOLUTION_NO_UPSCALE": 2048,
    "MIN_RESOLUTION_2X_UPSCALE": 1024,
    
    # Upscaling
    "UPSCALE_MODEL": "realesrgan-x4plus",
    "FACE_ENHANCE": True,
    "FACE_ENHANCE_MODEL": "CodeFormer",
    
    # Limits
    "MAX_POSTS": 0,  # 0 = unlimited
}
```

## 🔐 Instagram Authentication

### Option 1: Export cookies from browser (Recommended)

1. Install a cookie export extension (e.g., "Get cookies.txt" for Chrome)
2. Log into Instagram in your browser
3. Export cookies as JSON to `cookies.json`

### Option 2: Manual cookie file

Create `cookies.json` with your session cookies:

```json
[
  {
    "name": "sessionid",
    "value": "YOUR_SESSION_ID",
    "domain": ".instagram.com",
    "path": "/"
  },
  {
    "name": "csrftoken",
    "value": "YOUR_CSRF_TOKEN",
    "domain": ".instagram.com",
    "path": "/"
  }
]
```

## 🚀 Usage

### CLI Mode

```bash
python main.py
```

### Web GUI Mode

```bash
python main.py --gui
# Open http://localhost:8000
```

### Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║   ██╗      ██████╗ ██████╗  █████╗     ██████╗ ██╗██████╗ ███████╗║
║   ██║     ██╔═══██╗██╔══██╗██╔══██╗    ██╔══██╗██║██╔══██╗██╔════╝║
║   ██║     ██║   ██║██████╔╝███████║    ██████╔╝██║██████╔╝█████╗  ║
║   ██║     ██║   ██║██╔══██╗██╔══██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ║
║   ███████╗╚██████╔╝██║  ██║██║  ██║    ██║     ██║██║     ███████╗║
║   ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝║
╚══════════════════════════════════════════════════════════════════╝

============================================================
PIPELINE SUMMARY
============================================================
  Downloaded images:    39
  Duplicates removed:   19
  Extracted frames:     10
  Filtered (no person): 1
  Upscaled 2x:          19
  Final dataset:        19 images
  Total time:           0:02:54
============================================================
```

## 📁 Directory Structure

```
lora-dataset-pipeline/
├── main.py                 # Main pipeline orchestrator
├── gui.py                  # Web GUI (FastAPI)
├── requirements.txt        # Python dependencies
├── cookies.json            # Instagram session (gitignored)
├── modules/
│   ├── harvester.py        # Instagram scraping
│   ├── deduplicator.py     # Duplicate removal
│   ├── butcher.py          # Video frame extraction
│   ├── bouncer.py          # AI person filtering
│   ├── judge.py            # Quality analysis
│   ├── polisher.py         # Upscaling
│   └── utils.py            # Shared utilities
├── raw_downloads/          # Downloaded content (gitignored)
├── extracted_frames/       # Frames from videos (gitignored)
├── pending_upscale/        # Queue for upscaling (gitignored)
└── final_dataset/          # Output folder (gitignored)
```

## 🎯 Target Platform

Optimized and tested for:

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 7900X |
| **GPU** | AMD Radeon RX 7800 XT (16GB VRAM) |
| **OS** | Arch Linux |
| **Backend** | ROCm / Vulkan |

Also works on:
- macOS (Apple Silicon M1/M2) - with fallback upscaling
- Windows with NVIDIA GPU (CUDA)

## 🤖 Use Case: LoRA Training

This pipeline is designed to prepare datasets for training **LoRA (Low-Rank Adaptation)** models for Stable Diffusion.

### Recommended workflow:

1. **Scrape** target profile with this pipeline
2. **Caption** images with [BLIP](https://github.com/salesforce/BLIP) or [WD Tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)
3. **Train** with [Kohya_ss](https://github.com/bmaltais/kohya_ss) or [LoRA Easy Training Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts)
4. **Inference** with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This tool is for educational and personal use only. Respect Instagram's Terms of Service and the privacy of content creators. Always obtain proper permissions before using someone's likeness for AI training.

---

<div align="center">

**Made with ❤️ for the AI art community**

</div>
