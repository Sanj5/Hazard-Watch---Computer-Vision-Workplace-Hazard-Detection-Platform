import argparse
import time

import cv2

from src.config import AppConfig
from src.inference.adapters import YOLODetectorAdapter
from src.pipeline.safety_pipeline import SafetyPipeline


def draw_overlay(frame, analysis):
    lines = [
        f"Risk Score: {analysis.risk_score:.2f}",
        f"Risk Level: {analysis.risk_level}",
        f"Pred Hazard: {analysis.predicted_hazard_probability:.2f}",
    ]
    for i, msg in enumerate(analysis.coaching_messages):
        lines.append(f"Coach: {msg}")

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        y += 28


def parse_video_source(video_arg: str):
    if video_arg.isdigit():
        return int(video_arg)
    return video_arg


def main():
    parser = argparse.ArgumentParser(description="Industrial safety vision starter")
    parser.add_argument("--video", type=str, default="0", help="Webcam index (e.g., 0) or video file path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="YOLO tracker config")
    parser.add_argument("--no-track", action="store_true", help="Disable YOLO tracking IDs")
    args = parser.parse_args()

    source = parse_video_source(args.video)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video}")

    pipeline = SafetyPipeline(AppConfig())
    detector = YOLODetectorAdapter(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_tracking=not args.no_track,
        tracker=args.tracker,
    )

    last_ts = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(1e-3, now - last_ts)
        last_ts = now

        detections = detector.predict(frame)
        analysis = pipeline.analyze_frame(detections, dt, frame=frame)
        draw_overlay(frame, analysis)

        cv2.imshow("Safety Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
