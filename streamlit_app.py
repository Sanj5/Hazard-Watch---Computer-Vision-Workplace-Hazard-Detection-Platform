import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from src.config import AppConfig
from src.inference.adapters import YOLODetectorAdapter
from src.pipeline.safety_pipeline import SafetyPipeline


def draw_boxes(frame: np.ndarray, detections):
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        color = (0, 220, 100) if d.label == "person" else (255, 180, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{d.label} #{d.track_id} {d.confidence:.2f}"
        cv2.putText(frame, tag, (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_status(frame: np.ndarray, analysis):
    risk_color = (0, 255, 0)
    if analysis.risk_level == "MEDIUM":
        risk_color = (0, 220, 255)
    if analysis.risk_level == "HIGH":
        risk_color = (0, 0, 255)

    cv2.putText(frame, f"Risk {analysis.risk_score:.2f} ({analysis.risk_level})", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
    cv2.putText(frame, f"Pred Hazard {analysis.predicted_hazard_probability:.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if analysis.fatigue_events:
        top = max(analysis.fatigue_events, key=lambda f: f.fatigue_score)
        cv2.putText(
            frame,
            f"Fatigue worker #{top.track_id}: {top.fatigue_score:.2f}",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )


def resolve_video_source(source_mode: str, uploaded_file):
    if source_mode == "Webcam":
        return 0
    if uploaded_file is None:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def app():
    st.set_page_config(page_title="Workplace Hazard Detection", layout="wide")
    st.title("Computer Vision Workplace Hazard Detection and Prediction")

    # Keep UI minimal: fixed runtime defaults, no parameter controls.
    model_path = "yolov8n.pt"
    conf = 0.35
    iou = 0.45
    tracker_cfg = "bytetrack.yaml"
    fatigue_stationary_seconds = 30.0
    fatigue_movement_threshold = 6.0
    fatigue_alert_score = 0.65
    frame_limit = 300

    source_mode = st.radio("Video Source", options=["Webcam", "Upload Video"], horizontal=True)
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"]) if source_mode == "Upload Video" else None
    video_source = resolve_video_source(source_mode, uploaded)

    run_btn = st.button("Start Monitoring")

    col1, col2 = st.columns([2.0, 1.0])
    frame_slot = col1.empty()
    risk_slot = col2.empty()
    events_slot = col2.empty()
    fatigue_slot = col2.empty()
    coach_slot = col2.empty()

    if not run_btn:
        st.info("Click Start Monitoring.")
        return

    if video_source is None:
        st.error("Please upload a video file.")
        return

    config = AppConfig()
    config.thresholds.fatigue_stationary_seconds = float(fatigue_stationary_seconds)
    config.thresholds.fatigue_movement_threshold_px_s = float(fatigue_movement_threshold)
    config.thresholds.fatigue_alert_score = float(fatigue_alert_score)
    pipeline = SafetyPipeline(config)
    if not pipeline.pose_behavior.enabled:
        st.warning(
            "MediaPipe posture analysis is disabled because this environment does not provide `mediapipe.solutions`. "
            "All other detection, tracking, risk, and alert features continue to work."
        )
    detector = YOLODetectorAdapter(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        use_tracking=True,
        tracker=tracker_cfg,
    )

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Could not open selected video source.")
        return

    risk_history = []
    last_ts = time.time()
    processed = 0
    while cap.isOpened() and processed < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = max(1e-3, now - last_ts)
        last_ts = now

        detections = detector.predict(frame)
        analysis = pipeline.analyze_frame(detections, dt, frame=frame)

        render = frame.copy()
        draw_boxes(render, detections)
        draw_status(render, analysis)
        frame_slot.image(cv2.cvtColor(render, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        risk_history.append(analysis.risk_score)
        risk_slot.line_chart({"risk_score": risk_history[-100:]})

        with events_slot.container():
            st.markdown("### Event Counters")
            c1, c2 = st.columns(2)
            c1.metric("PPE Violations", len(analysis.ppe_violations))
            c2.metric("Unsafe Behaviors", len(analysis.behavior_events))
            c1.metric("Near Misses", len(analysis.near_miss_events))
            c2.metric("Fatigue Alerts", len(analysis.fatigue_events))
            st.metric("Scene Hazards", len(analysis.scene_hazard_events))

        with fatigue_slot.container():
            st.markdown("### Fatigue Alerts")
            if analysis.fatigue_events:
                for f in analysis.fatigue_events[:3]:
                    st.warning(
                        f"Worker #{f.track_id} fatigue={f.fatigue_score:.2f}, "
                        f"stationary={f.stationary_seconds:.1f}s"
                    )
            else:
                st.caption("No fatigue alerts in current frame.")

        with coach_slot.container():
            st.markdown("### Real-time Safety Coaching")
            for msg in analysis.coaching_messages:
                st.warning(msg)

        processed += 1

    cap.release()
    st.success(f"Monitoring complete. Processed {processed} frames.")


if __name__ == "__main__":
    app()
