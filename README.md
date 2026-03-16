# Computer Vision Workplace Hazard Detection Platform

This project is a practical starter for building a computer vision safety platform with:

- PPE detection
- Behavior monitoring
- Worker fatigue analysis
- Surroundings hazard analysis
- Hazard prediction
- Near-miss detection
- Risk score calculation
- Real-time safety coaching
- Future collision detection

## 1) What this starter gives you

- A modular Python codebase (`src/`) for real-time frame-by-frame safety analysis.
- YOLOv8/YOLOv9-ready detection adapter with ByteTrack/BOTSort tracker support.
- MediaPipe-based posture cue detection hook.
- PyTorch LSTM trajectory forecasting wrapper for future collision prediction.
- Dynamic risk scoring and real-time safety coaching.
- Streamlit dashboard for monitoring and alerts.
- A training script template and unit tests.

This is designed so you can plug in your own detector model (YOLO/RT-DETR/etc.) and progressively improve accuracy.

## 2) Technology stack

- Python, PyTorch, OpenCV
- Object detection: YOLOv8/YOLOv9 (`person`, `helmet`, `vest`, `gloves`, `goggles`, `boots`, `forklift`, `truck`, etc.)
- Pose estimation: MediaPipe (implemented) or OpenPose (plug-in option)
- Fatigue analysis: temporal low-movement and posture-cue scoring (implemented)
- Surroundings analysis: scene hazards, equipment congestion, and worker-equipment proximity (implemented)
- Tracking: ByteTrack/BOTSort through Ultralytics tracker config (DeepSORT can be integrated as adapter)
- Trajectory prediction: LSTM (implemented wrapper) or Transformer (recommended extension)
- Hazard prediction: temporal risk model (starter) with upgrade path to sequence models

## 3) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4) Run (YOLO pipeline)

```powershell
python -m src.main --video 0
```

Use `--video <path>` for a file.

Custom model example:

```powershell
python -m src.main --video 0 --model path\to\best.pt --conf 0.4 --iou 0.5
```

Disable tracking (not recommended for collision prediction):

```powershell
python -m src.main --video 0 --no-track
```

## 5) Run Streamlit platform

```powershell
streamlit run streamlit_app.py
```

Dashboard includes:

- Live frame visualization with detections
- Risk score trend chart
- PPE / behavior / near-miss counters
- Real-time safety coaching alerts

## 6) Detection output format

The YOLO adapter in `src/inference/adapters.py` converts detections to:

```python
{
    "track_id": 12,
    "label": "person",
    "confidence": 0.91,
    "bbox": [x1, y1, x2, y2]
}
```

Then feed those detections to the pipeline.

## 7) Suggested project structure

```text
.
|-- streamlit_app.py
|-- requirements.txt
|-- scripts/
|   |-- train_hazard_model.py
|-- src/
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
```

## 8) Dataset suggestions

- PPE datasets:
    - Roboflow PPE datasets (hard hats, vests, gloves, goggles)
    - CHV / construction safety datasets from Kaggle
- Behavior datasets:
    - UCF-Crime (unsafe behavior context)
    - HMDB51 / Kinetics subsets for action priors
- Industrial assets / vehicles:
    - Custom warehouse/factory annotations for forklifts, trucks, robots
- Near-miss and incident data:
    - Site CCTV clips + incident logs with timestamps and zones

Recommended annotation schema:

- Frame-level boxes + IDs: `person`, `helmet`, `vest`, `gloves`, `goggles`, `boots`, `forklift`, `truck`, `robot`
- Event labels: `running_restricted`, `phone_use`, `unsafe_climbing`, `unsafe_posture`, `near_miss`
- Trajectory windows: `(track_id, t, x, y, vx, vy)`
- Scene metadata: restricted zones, hazard zones, shift, weather/lighting

## 9) Training pipeline (recommended)

1. Train/fine-tune YOLO PPE + equipment detector on site-specific images.
2. Train behavior classifier from pose/action features.
3. Train LSTM/Transformer trajectory forecaster on tracked center-point sequences.
4. Train hazard prediction model using aggregated temporal features.
5. Calibrate risk weights/thresholds with historical incidents.

Starter scripts provided:

- `scripts/train_hazard_model.py` for baseline hazard model pattern.
- Add `scripts/train_trajectory_lstm.py` and `scripts/train_ppe_yolo.py` for production training.

## 10) Evaluation metrics

- Detection (YOLO): mAP@0.5, mAP@0.5:0.95, precision, recall
- Tracking: MOTA, IDF1, ID switches, HOTA
- Behavior detection: F1-score, precision/recall by class
- Near-miss and collision prediction: lead-time accuracy, false alarm rate, miss rate
- Hazard prediction: AUROC, PR-AUC, calibration error
- End-to-end: alert latency (ms), FPS throughput, operator response acceptance rate

## 11) Deployment approach

- Edge inference node (GPU): on-site processing for low latency
- Message broker (MQTT/Kafka): stream events to central safety service
- Alert channels: dashboard, siren, SMS/Teams, control room API
- Storage: event clips + metadata in object storage and SQL/Timeseries DB
- MLOps: model registry, staged rollout, drift monitoring, periodic retraining

Reference latency budget:

- Detection + tracking: 25-40 ms/frame
- Behavior + trajectory + risk scoring: 5-15 ms/frame
- End-to-end alerting target: < 250 ms

## 12) Suggested roadmap

1. Collect and label your site-specific PPE + hazard data.
2. Train/fine-tune PPE detector classes.
3. Add robust tracking and trajectory smoothing.
4. Train hazard prediction on temporal windows.
5. Calibrate risk thresholds with real incident history.
6. Deploy with edge GPU + alert channels.

## 13) Safety note

This is a decision-support system, not a replacement for human safety protocols.
