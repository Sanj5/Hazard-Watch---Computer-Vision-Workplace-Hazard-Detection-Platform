import tempfile
import time
import json
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.config import AppConfig
from src.inference.adapters import YOLODetectorAdapter
from src.pipeline.safety_pipeline import SafetyPipeline


def inject_ui_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(22, 163, 74, 0.16), transparent 36%),
                radial-gradient(circle at 100% 20%, rgba(14, 116, 144, 0.14), transparent 36%),
                #f7faf9;
            font-family: 'DM Sans', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        .hw-hero {
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.9rem;
            background: linear-gradient(120deg, #0f172a 0%, #0b3a44 60%, #14532d 100%);
            color: #f8fafc;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.25);
        }

        .hw-subtle {
            color: rgba(248, 250, 252, 0.86);
            margin-top: 0.2rem;
        }

        .hw-chip {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.26);
            background: rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            padding: 0.2rem 0.62rem;
            margin: 0.42rem 0.4rem 0 0;
            font-size: 0.8rem;
            font-weight: 600;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(180deg, #f8fbfa 0%, #eef5f3 100%);
        }

        [data-testid="stMetricValue"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stButton > button {
            border-radius: 12px;
            border: 0;
            background: linear-gradient(120deg, #0f766e 0%, #15803d 100%);
            color: #ffffff;
            font-weight: 700;
            min-height: 44px;
        }

        .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
            transition: all 0.2s ease;
        }

        .hw-empty {
            border: 1px dashed rgba(15, 23, 42, 0.2);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.7);
            padding: 1.1rem;
            color: #334155;
        }

        @media (max-width: 900px) {
            .hw-hero {
                padding: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_level_color(risk_level: str) -> str:
    if risk_level == "HIGH":
        return "#dc2626"
    if risk_level == "MEDIUM":
        return "#d97706"
    return "#16a34a"


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


def _zone_id_from_point(
    x: float,
    y: float,
    frame_w: int,
    frame_h: int,
    rows: int = 3,
    cols: int = 3,
) -> Tuple[int, int]:
    col = int(np.clip((x / max(frame_w, 1)) * cols, 0, cols - 1))
    row = int(np.clip((y / max(frame_h, 1)) * rows, 0, rows - 1))
    return row, col


def update_zone_risk(
    zone_scores: Dict[Tuple[int, int], float],
    zone_hits: Dict[Tuple[int, int], int],
    detections,
    risk_score: float,
    frame_shape,
    rows: int = 3,
    cols: int = 3,
):
    frame_h, frame_w = frame_shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        zone = _zone_id_from_point(center_x, center_y, frame_w, frame_h, rows=rows, cols=cols)
        weighted_risk = float(risk_score) * max(float(det.confidence), 0.25)
        zone_scores[zone] = zone_scores.get(zone, 0.0) + weighted_risk
        zone_hits[zone] = zone_hits.get(zone, 0) + 1


def render_zone_heatmap_image(
    zone_scores: Dict[Tuple[int, int], float],
    zone_hits: Dict[Tuple[int, int], int],
    rows: int = 3,
    cols: int = 3,
) -> np.ndarray:
    avg = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            hits = zone_hits.get((r, c), 0)
            if hits > 0:
                avg[r, c] = zone_scores.get((r, c), 0.0) / hits

    norm = avg.copy()
    peak = float(np.max(norm))
    if peak > 0:
        norm = norm / peak

    heat = (norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat_color = cv2.resize(heat_color, (360, 360), interpolation=cv2.INTER_NEAREST)

    cell_h = heat_color.shape[0] // rows
    cell_w = heat_color.shape[1] // cols
    for r in range(rows):
        for c in range(cols):
            x = c * cell_w
            y = r * cell_h
            cv2.rectangle(heat_color, (x, y), (x + cell_w, y + cell_h), (255, 255, 255), 1)
            value = avg[r, c]
            label = f"Z{r + 1}-{c + 1}: {value:.2f}"
            cv2.putText(
                heat_color,
                label,
                (x + 6, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)


def speak_coaching_message(message: str):
    safe_message = json.dumps(message)
    components.html(
        f"""
        <script>
        const msg = {safe_message};
        if (msg && window.speechSynthesis) {{
            const synth = window.speechSynthesis;
            // Replace current utterance if any; Python-side cooldown throttles re-triggers.
            synth.cancel();
            const u = new SpeechSynthesisUtterance(msg);
            u.rate = 1.0;
            u.pitch = 1.0;
            u.volume = 1.0;
            synth.speak(u);
        }}
        </script>
        """,
        height=0,
        width=0,
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
    st.set_page_config(
        page_title="HazardWatch Live",
        page_icon="⛑️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_ui_styles()

    st.markdown(
        """
        <div class="hw-hero">
            <h2 style="margin:0;">HazardWatch Command Center</h2>
            <p class="hw-subtle">Live workplace risk intelligence powered by object detection, tracking, and safety coaching.</p>
            <span class="hw-chip">PPE Monitoring</span>
            <span class="hw-chip">Near-Miss Detection</span>
            <span class="hw-chip">Fatigue Alerts</span>
            <span class="hw-chip">Real-time Coaching</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep UI minimal: fixed runtime defaults, no parameter controls.
    model_path = "yolov8n.pt"
    conf = 0.35
    iou = 0.45
    tracker_cfg = "bytetrack.yaml"
    fatigue_stationary_seconds = 30.0
    fatigue_movement_threshold = 6.0
    fatigue_alert_score = 0.65
    frame_limit = 300

    with st.sidebar:
        st.markdown("### Monitoring Setup")
        source_mode = st.radio("Video Source", options=["Webcam", "Upload Video"], horizontal=True)
        uploaded = (
            st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
            if source_mode == "Upload Video"
            else None
        )
        st.caption("Runtime uses tuned defaults for stable real-time behavior.")
        voice_coach_enabled = st.toggle("Voice Coach", value=True)
        voice_cooldown_seconds = st.slider("Voice Cooldown (seconds)", min_value=3, max_value=20, value=7, step=1)
        run_btn = st.button("Start Monitoring", use_container_width=True)

    video_source = resolve_video_source(source_mode, uploaded)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Pipeline", "Ready")
    kpi2.metric("Model", "yolov8n")
    kpi3.metric("Tracking", "ByteTrack")
    kpi4.metric("Frame Limit", str(frame_limit))

    col1, col2 = st.columns([2.2, 1.0])
    frame_slot = col1.empty()
    risk_head = col2.empty()
    risk_slot = col2.empty()
    events_slot = col2.container(border=True)
    fatigue_slot = col2.container(border=True)
    coach_slot = col2.container(border=True)
    heatmap_slot = col2.container(border=True)
    progress_slot = col1.empty()

    with events_slot:
        st.markdown("### Event Counters")
        ev_col1, ev_col2 = st.columns(2)
        ppe_metric = ev_col1.empty()
        unsafe_metric = ev_col2.empty()
        near_miss_metric = ev_col1.empty()
        fatigue_metric = ev_col2.empty()
        scene_hazard_metric = st.empty()

    with fatigue_slot:
        st.markdown("### Fatigue Alerts")
        fatigue_body = st.empty()

    with coach_slot:
        st.markdown("### Real-time Safety Coaching")
        coach_body = st.empty()

    with heatmap_slot:
        st.markdown("### Zone Risk Heatmap")
        heatmap_image_slot = st.empty()
        zone_summary_slot = st.empty()

    if not run_btn:
        frame_slot.markdown(
            """
            <div class="hw-empty">
                <h4 style="margin-top:0;">Ready to monitor</h4>
                <p style="margin-bottom:0.45rem;">Select a source in the sidebar and start monitoring to view live detections and risk analytics.</p>
                <p style="margin-bottom:0;">Tip: Upload a short clip for a quick demo run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
    zone_scores: Dict[Tuple[int, int], float] = {}
    zone_hits: Dict[Tuple[int, int], int] = {}
    last_voice_message = ""
    last_voice_ts = 0.0
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
        update_zone_risk(zone_scores, zone_hits, detections, analysis.risk_score, frame.shape)

        render = frame.copy()
        draw_boxes(render, detections)
        draw_status(render, analysis)
        frame_slot.image(cv2.cvtColor(render, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        progress = min(1.0, (processed + 1) / max(frame_limit, 1))
        progress_slot.progress(progress, text=f"Frame {processed + 1} / {frame_limit}")

        risk_history.append(analysis.risk_score)
        risk_head.markdown(
            f"### Current Risk: <span style='color:{risk_level_color(analysis.risk_level)}'>{analysis.risk_level}</span>",
            unsafe_allow_html=True,
        )
        risk_slot.line_chart({"risk_score": risk_history[-100:]})

        ppe_metric.metric("PPE Violations", len(analysis.ppe_violations))
        unsafe_metric.metric("Unsafe Behaviors", len(analysis.behavior_events))
        near_miss_metric.metric("Near Misses", len(analysis.near_miss_events))
        fatigue_metric.metric("Fatigue Alerts", len(analysis.fatigue_events))
        scene_hazard_metric.metric("Scene Hazards", len(analysis.scene_hazard_events))

        with fatigue_body.container():
            if analysis.fatigue_events:
                for f in analysis.fatigue_events[:3]:
                    st.warning(
                        f"Worker #{f.track_id} fatigue={f.fatigue_score:.2f}, "
                        f"stationary={f.stationary_seconds:.1f}s"
                    )
            else:
                st.caption("No fatigue alerts in current frame.")

        with coach_body.container():
            for msg in analysis.coaching_messages:
                st.warning(msg)
            if not analysis.coaching_messages:
                st.caption("No coaching prompts in current frame.")

        heatmap_image = render_zone_heatmap_image(zone_scores, zone_hits)
        heatmap_image_slot.image(heatmap_image, channels="RGB", use_container_width=True)

        zone_rank = []
        for (r, c), score in zone_scores.items():
            hits = zone_hits.get((r, c), 0)
            if hits > 0:
                zone_rank.append((score / hits, f"Z{r + 1}-{c + 1}"))
        zone_rank.sort(reverse=True)
        if zone_rank:
            top_zones = ", ".join([f"{name} ({val:.2f})" for val, name in zone_rank[:3]])
            zone_summary_slot.caption(f"Highest-risk zones: {top_zones}")
        else:
            zone_summary_slot.caption("No zone risk data yet.")

        if voice_coach_enabled and analysis.coaching_messages:
            speak_now = (
                analysis.risk_level in {"HIGH", "MEDIUM"}
                and "Safety status normal" not in analysis.coaching_messages[0]
            )
            if speak_now:
                candidate = analysis.coaching_messages[0]
                elapsed = now - last_voice_ts
                # Enforce cooldown regardless of message changes to avoid repeated speech on every frame.
                if elapsed >= float(voice_cooldown_seconds):
                    speak_coaching_message(candidate)
                    last_voice_message = candidate
                    last_voice_ts = now

        processed += 1

    cap.release()
    st.success(f"Monitoring complete. Processed {processed} frames.")


if __name__ == "__main__":
    app()
