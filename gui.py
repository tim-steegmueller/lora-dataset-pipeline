"""
Web GUI for LoRA Dataset Pipeline
Modern, responsive interface with live logs and progress tracking.
"""

import asyncio
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(title="LoRA Dataset Pipeline", version="1.0.0")

# Global state
pipeline_running = False
pipeline_thread: Optional[threading.Thread] = None
log_queue: queue.Queue = queue.Queue()
current_stats: dict = {}
connected_websockets: list[WebSocket] = []


# =============================================================================
# MODELS
# =============================================================================

class PipelineConfig(BaseModel):
    target_users: list[str]
    max_posts: int = 100
    download_posts: bool = True
    download_stories: bool = True
    enable_person_filter: bool = True
    min_person_ratio: float = 0.05


# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoRA Dataset Pipeline</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --bg-card: #16161f;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #2d2d3a;
            --glow: rgba(99, 102, 241, 0.3);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background */
        .bg-pattern {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(168, 85, 247, 0.05) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            position: relative;
            z-index: 1;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .logo {
            font-size: 3rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        /* Grid layout */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 1024px) {
            .grid { grid-template-columns: 1fr; }
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: var(--accent-primary);
            box-shadow: 0 0 30px var(--glow);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .card-icon {
            width: 40px;
            height: 40px;
            background: var(--accent-gradient);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
        }

        /* Form elements */
        .form-group {
            margin-bottom: 1.25rem;
        }

        label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 0.875rem 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--glow);
        }

        /* Checkbox */
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }

        .checkbox-item input {
            width: 18px;
            height: 18px;
            accent-color: var(--accent-primary);
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 1rem 2rem;
            border: none;
            border-radius: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: var(--accent-gradient);
            color: white;
            box-shadow: 0 4px 20px var(--glow);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px var(--glow);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .btn-danger {
            background: var(--error);
            color: white;
        }

        .btn-full {
            width: 100%;
        }

        /* Log console */
        .log-console {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 12px;
            height: 400px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            padding: 1rem;
        }

        .log-line {
            padding: 0.25rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            white-space: pre-wrap;
            word-break: break-all;
        }

        .log-line.info { color: var(--text-secondary); }
        .log-line.warning { color: var(--warning); }
        .log-line.error { color: var(--error); }
        .log-line.success { color: var(--success); }

        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }

        .stat-item {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }

        /* Progress bar */
        .progress-container {
            margin: 1.5rem 0;
        }

        .progress-bar {
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent-gradient);
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .progress-text {
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }

        /* Status indicator */
        .status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-idle { background: var(--bg-tertiary); }
        .status-idle .status-dot { background: var(--text-muted); animation: none; }

        .status-running { background: rgba(99, 102, 241, 0.2); }
        .status-running .status-dot { background: var(--accent-primary); }

        .status-complete { background: rgba(16, 185, 129, 0.2); }
        .status-complete .status-dot { background: var(--success); animation: none; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Image preview grid */
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 0.75rem;
            max-height: 300px;
            overflow-y: auto;
        }

        .preview-item {
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid transparent;
            transition: all 0.2s ease;
        }

        .preview-item:hover {
            border-color: var(--accent-primary);
            transform: scale(1.05);
        }

        .preview-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-primary);
        }
    </style>
</head>
<body>
    <div class="bg-pattern"></div>

    <div class="container">
        <header>
            <h1 class="logo">LoRA Pipeline</h1>
            <p class="subtitle">Instagram Dataset Automation</p>
        </header>

        <div class="grid">
            <!-- Config Panel -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">⚙️</div>
                    <h2 class="card-title">Configuration</h2>
                </div>

                <div class="form-group">
                    <label>Target Username(s)</label>
                    <input type="text" id="targetUsers" placeholder="username1, username2" value="">
                </div>

                <div class="form-group">
                    <label>Max Posts per Profile</label>
                    <input type="number" id="maxPosts" value="100" min="1" max="1000">
                </div>

                <div class="form-group">
                    <label>Options</label>
                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" id="downloadPosts" checked>
                            <span>Posts</span>
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" id="downloadStories" checked>
                            <span>Stories</span>
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" id="personFilter" checked>
                            <span>Person Filter</span>
                        </label>
                    </div>
                </div>

                <div class="form-group">
                    <label>Min Person Size (%)</label>
                    <input type="number" id="minPersonRatio" value="5" min="1" max="50">
                </div>

                <button class="btn btn-primary btn-full" id="startBtn" onclick="startPipeline()">
                    🚀 Start Pipeline
                </button>
            </div>

            <!-- Status Panel -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <h2 class="card-title">Status</h2>
                    <div class="status status-idle" id="statusBadge">
                        <span class="status-dot"></span>
                        <span id="statusText">Idle</span>
                    </div>
                </div>

                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>
                    <div class="progress-text">
                        <span id="progressStage">Ready</span>
                        <span id="progressPercent">0%</span>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="statImages">0</div>
                        <div class="stat-label">Images</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statVideos">0</div>
                        <div class="stat-label">Videos</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statFrames">0</div>
                        <div class="stat-label">Frames</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statFiltered">0</div>
                        <div class="stat-label">Filtered</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statFinal">0</div>
                        <div class="stat-label">Final</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statTime">0s</div>
                        <div class="stat-label">Time</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Log Console -->
        <div class="card" style="margin-top: 1.5rem;">
            <div class="card-header">
                <div class="card-icon">📝</div>
                <h2 class="card-title">Live Logs</h2>
            </div>
            <div class="log-console" id="logConsole">
                <div class="log-line info">Pipeline ready. Configure settings and click Start.</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let startTime = null;
        let timerInterval = null;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                addLog('Connected to server', 'success');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'log') {
                    addLog(data.message, data.level || 'info');
                } else if (data.type === 'stats') {
                    updateStats(data.stats);
                } else if (data.type === 'status') {
                    updateStatus(data.status);
                } else if (data.type === 'progress') {
                    updateProgress(data.stage, data.percent);
                }
            };

            ws.onclose = () => {
                addLog('Disconnected from server', 'warning');
                setTimeout(connectWebSocket, 3000);
            };
        }

        function addLog(message, level = 'info') {
            const console = document.getElementById('logConsole');
            const line = document.createElement('div');
            line.className = `log-line ${level}`;
            line.textContent = message;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }

        function updateStats(stats) {
            document.getElementById('statImages').textContent = stats.downloaded_images || 0;
            document.getElementById('statVideos').textContent = stats.downloaded_videos || 0;
            document.getElementById('statFrames').textContent = stats.extracted_frames || 0;
            document.getElementById('statFiltered').textContent =
                (stats.filtered_no_person || 0) + (stats.filtered_person_small || 0);
            document.getElementById('statFinal').textContent = stats.final_count || 0;
        }

        function updateStatus(status) {
            const badge = document.getElementById('statusBadge');
            const text = document.getElementById('statusText');
            const btn = document.getElementById('startBtn');

            badge.className = 'status status-' + status;
            text.textContent = status.charAt(0).toUpperCase() + status.slice(1);

            if (status === 'running') {
                btn.disabled = true;
                btn.textContent = '⏳ Running...';
                startTimer();
            } else {
                btn.disabled = false;
                btn.textContent = '🚀 Start Pipeline';
                stopTimer();
            }
        }

        function updateProgress(stage, percent) {
            document.getElementById('progressFill').style.width = percent + '%';
            document.getElementById('progressStage').textContent = stage;
            document.getElementById('progressPercent').textContent = percent + '%';
        }

        function startTimer() {
            startTime = Date.now();
            timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                document.getElementById('statTime').textContent =
                    mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
            }, 1000);
        }

        function stopTimer() {
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
            }
        }

        async function startPipeline() {
            const config = {
                target_users: document.getElementById('targetUsers').value
                    .split(',').map(s => s.trim()).filter(s => s),
                max_posts: parseInt(document.getElementById('maxPosts').value) || 100,
                download_posts: document.getElementById('downloadPosts').checked,
                download_stories: document.getElementById('downloadStories').checked,
                enable_person_filter: document.getElementById('personFilter').checked,
                min_person_ratio: (parseInt(document.getElementById('minPersonRatio').value) || 5) / 100
            };

            if (config.target_users.length === 0) {
                addLog('Please enter at least one username', 'error');
                return;
            }

            try {
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });

                const result = await response.json();

                if (result.success) {
                    addLog('Pipeline started!', 'success');
                } else {
                    addLog('Failed to start: ' + result.error, 'error');
                }
            } catch (error) {
                addLog('Error: ' + error.message, 'error');
            }
        }

        // Initialize
        connectWebSocket();
    </script>
</body>
</html>
"""


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)

    try:
        while True:
            # Check for new logs
            while not log_queue.empty():
                log_msg = log_queue.get_nowait()
                await broadcast({"type": "log", "message": log_msg, "level": "info"})

            # Send stats update
            if current_stats:
                await websocket.send_json({"type": "stats", "stats": current_stats})

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


async def broadcast(message: dict):
    """Broadcast message to all connected clients."""
    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except:
            pass


@app.post("/api/start")
async def start_pipeline(config: PipelineConfig):
    global pipeline_running, pipeline_thread, current_stats

    if pipeline_running:
        return {"success": False, "error": "Pipeline already running"}

    # Update config
    from main import CONFIG
    CONFIG["TARGET_USERS"] = config.target_users
    CONFIG["MAX_POSTS"] = config.max_posts
    CONFIG["DOWNLOAD_POSTS"] = config.download_posts
    CONFIG["DOWNLOAD_STORIES"] = config.download_stories
    CONFIG["ENABLE_PERSON_FILTER"] = config.enable_person_filter
    CONFIG["MIN_PERSON_RATIO"] = config.min_person_ratio

    # Start pipeline in background thread
    def run():
        global pipeline_running, current_stats
        pipeline_running = True

        try:
            from main import run_pipeline

            def log_callback(msg):
                log_queue.put(msg)

            current_stats = run_pipeline(CONFIG, log_callback)
        except Exception as e:
            log_queue.put(f"ERROR: {str(e)}")
        finally:
            pipeline_running = False

    pipeline_thread = threading.Thread(target=run)
    pipeline_thread.start()

    return {"success": True}


@app.get("/api/status")
async def get_status():
    return {
        "running": pipeline_running,
        "stats": current_stats
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_gui(config: dict):
    """Start the GUI server."""
    print("\n" + "=" * 60)
    print("  LoRA Dataset Pipeline - Web GUI")
    print("  Open in browser: http://localhost:8000")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    from main import CONFIG
    run_gui(CONFIG)

