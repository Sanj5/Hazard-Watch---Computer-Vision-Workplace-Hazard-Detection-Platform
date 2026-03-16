# System Design: Workplace Hazard Detection and Prediction

## Objectives

- Detect PPE violations in real time.
- Detect unsafe worker behavior from objects and pose cues.
- Detect near-misses and predict future collisions.
- Predict hazard escalation and generate a dynamic risk score.
- Deliver real-time safety coaching alerts.

## Online Pipeline

1. Frame ingestion from CCTV/RTSP/video file.
2. YOLOv8/YOLOv9 object detection (`person`, PPE, vehicles/equipment).
3. Multi-object tracking with ByteTrack/BOTSort (DeepSORT adapter optional).
4. Pose behavior analysis (MediaPipe/OpenPose).
5. Near-miss + trajectory-based collision prediction.
6. Hazard probability estimation from temporal risk features.
7. Dynamic risk scoring and coaching output.
8. UI + alerts (Streamlit dashboard, API/webhook integration).

## Core Models

- Detection: Ultralytics YOLO adapter in `src/inference/adapters.py`.
- Pose cues: MediaPipe analyzer in `src/pose/mediapipe_pose.py`.
- Collision prediction: LSTM wrapper in `src/predictors/trajectory_lstm.py` and forecast fusion in `src/predictors/collision_predictor.py`.
- Hazard prediction: Temporal smoothing baseline in `src/predictors/hazard_predictor.py`.

## Risk Score Logic

Dynamic score combines:

- PPE non-compliance
- Unsafe behavior severity
- Near-miss confidence
- Collision risk confidence
- Hazard prediction probability

Weights and thresholds are configurable in `src/config.py`.

## Evaluation Strategy

- Detection: mAP, precision, recall
- Tracking: IDF1, HOTA, MOTA
- Behavior: per-class precision/recall/F1
- Prediction: AUROC/PR-AUC and lead-time for collisions
- Runtime: FPS, p95 inference latency, end-to-end alert delay

## Deployment Blueprint

- Edge GPU worker for low-latency inference.
- Message bus for events and alerts.
- Event store with video snippet retention policy.
- Monitoring for model drift and false alarm trends.
- Scheduled retraining with incident-verified labels.
