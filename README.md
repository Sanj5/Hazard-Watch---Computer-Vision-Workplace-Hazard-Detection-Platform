# Hazard Watch: Computer Vision Workplace Hazard Detection Platform

A practical, modular safety analytics project for real-time workplace monitoring using computer vision.

Hazard Watch combines object detection, tracking, behavior signals, fatigue monitoring, hazard prediction, and real-time coaching into one pipeline that can run from webcam or video input.

## Table of Contents

- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Run the Platform](#run-the-platform)
- [Dashboard Experience](#dashboard-experience)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Configuration Notes](#configuration-notes)
- [Testing](#testing)
- [Git and GitHub Workflow](#git-and-github-workflow)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Safety Disclaimer](#safety-disclaimer)

## Overview

This project is designed as a strong starter for industrial safety intelligence systems.

It helps detect and reason about:

- Personal protective equipment compliance
- Unsafe worker behavior
- Fatigue-like patterns from temporal movement and posture cues
- Near-miss events
- Scene-level hazards and risk escalation
- Short-term hazard probability trends
- Real-time operator coaching prompts
- Voice-based safety coaching prompts
- Zone-wise risk heatmap trends

The codebase is intentionally modular so you can replace detectors, models, and thresholds with site-specific versions.

## Core Capabilities

- Real-time frame-by-frame analysis pipeline
- YOLO adapter with tracking support
- Pose and behavior integration hooks
- Temporal risk scoring and hazard prediction
- Near-miss logic and scene hazard checks
- Real-time coaching message generation
- Voice coach guidance for operators
- Zone-based risk heatmap analytics
- Streamlit dashboard with live frame, risk chart, and event panels

## Tech Stack

- Python
- OpenCV
- PyTorch
- Streamlit
- Ultralytics-style detector and tracker integration
- Optional MediaPipe pose signals

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- streamlit_app.py
|-- scripts/
|   |-- train_hazard_model.py
|-- src/
|   |-- main.py
|   |-- config.py
|   |-- coaching/
|   |-- detectors/
|   |-- inference/
|   |-- pipeline/
|   |-- pose/
|   |-- predictors/
|   |-- risk/
|   |-- tracking/
|   |-- utils/
|-- tests/
|   |-- test_risk_scorer.py
```

## Quick Start

### 1) Prerequisites

- Python 3.10 or newer recommended
- pip
- A compatible camera, or a test video file

### 2) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the Platform

### CLI pipeline

Use webcam:

```powershell
python -m src.main --video 0
```

Use a video file:

```powershell
python -m src.main --video path\to\video.mp4
```

Use a custom detector model:

```powershell
python -m src.main --video 0 --model path\to\best.pt --conf 0.4 --iou 0.5
```

Disable tracking (not recommended for trajectory and collision logic):

```powershell
python -m src.main --video 0 --no-track
```

### Streamlit dashboard

```powershell
streamlit run streamlit_app.py
```

## Dashboard Experience

The dashboard provides:

- Live annotated frame stream
- Risk trend chart
- Zone risk heatmap by monitored area
- Fixed Event Counters panel shown once, with values updated every frame
- Fatigue alert panel
- Real-time coaching panel
- Voice coach panel for spoken safety cues
- Processing progress indicator

## How the Pipeline Works

At a high level:

1. Video frame is read from webcam or uploaded file.
2. Detector adapter returns labeled bounding boxes and confidence.
3. Tracker assigns and maintains identities across frames.
4. Safety modules analyze behavior, PPE, near-miss, fatigue, surroundings, and temporal risk.
5. Aggregated analysis produces:
   - risk score and risk level
   - hazard probability estimate
   - event lists and coaching prompts
6. Results are rendered in the dashboard and charts.

Detection output consumed by the pipeline follows this shape:

```python
{
    "track_id": 12,
    "label": "person",
    "confidence": 0.91,
    "bbox": [x1, y1, x2, y2]
}
```

## Configuration Notes

Primary runtime controls include:

- Detection confidence threshold
- IoU threshold
- Tracker config
- Fatigue stationary time threshold
- Fatigue movement threshold
- Fatigue alert threshold
- Frame processing limit

Adjust values in the app or config modules based on camera placement, workcell geometry, and alert tolerance.

## Testing

Run tests:

```powershell
pytest -q
```

Current tests include risk scorer coverage and can be extended with detector, tracker, and pipeline integration tests.

## Git and GitHub Workflow

Clone:

```powershell
git clone <your-repo-url>
cd Hazard-Watch---Computer-Vision-Workplace-Hazard-Detection-Platform
```

Create a branch:

```powershell
git checkout -b feature/short-description
```

Commit changes:

```powershell
git add .
git commit -m "feat: add short summary"
```

Push branch:

```powershell
git push -u origin feature/short-description
```

Sync with main:

```powershell
git checkout main
git pull origin main
git checkout feature/short-description
git merge main
```

Then open a pull request from feature branch to main.

## Troubleshooting

- Webcam does not open:
  - Verify camera is not used by another app.
  - Try a video file source first.
- MediaPipe warning appears:
  - The app continues with core detection and risk modules.
  - Install a compatible mediapipe build if posture cues are required.
- Low FPS:
  - Use smaller video resolution.
  - Reduce frame limit.
  - Use GPU-enabled environment.
- Too many false alerts:
  - Recalibrate thresholds for your site conditions.
  - Improve detector quality with domain-specific training data.

## Roadmap

- Add richer temporal hazard models
- Add trajectory model training scripts
- Add configurable voice coach with multilingual prompts
- Add persistent zone risk heatmap with time-window filters
- Add alert export and event logging backends
- Add configuration presets per workplace type
- Add end-to-end benchmark and profiling suite

## Contributing

Contributions are welcome.

Recommended contribution flow:

1. Fork or branch from main.
2. Keep changes focused and small.
3. Add or update tests when behavior changes.
4. Open a pull request with:
   - clear summary
   - before and after behavior
   - screenshots for UI changes

## Safety Disclaimer

This software is decision-support only. It does not replace site safety standards, supervision, training, or regulatory compliance procedures.
