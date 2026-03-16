from typing import List, Optional

from src.utils.types import Detection

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class YOLODetectorAdapter:
    """Ultralytics YOLO adapter that emits pipeline Detection objects."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        use_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            )

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_tracking = use_tracking
        self.tracker = tracker
        self.last_result = None

    @staticmethod
    def _to_detection(track_id: int, label: str, conf: float, xyxy) -> Detection:
        x1, y1, x2, y2 = xyxy
        return Detection(
            track_id=track_id,
            label=label,
            confidence=float(conf),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
        )

    def _predict_once(self, frame):
        if self.use_tracking:
            return self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )
        return self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

    def predict(self, frame) -> List[Detection]:
        results = self._predict_once(frame)
        if not results:
            return []

        result = results[0]
        self.last_result = result
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        names = getattr(result, "names", {})
        detections: List[Detection] = []

        cls_values = boxes.cls.tolist() if boxes.cls is not None else []
        conf_values = boxes.conf.tolist() if boxes.conf is not None else []
        xyxy_values = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        id_values: Optional[List[int]] = None

        if hasattr(boxes, "id") and boxes.id is not None:
            id_values = [int(x) for x in boxes.id.tolist()]

        for idx, xyxy in enumerate(xyxy_values):
            cls_id = int(cls_values[idx]) if idx < len(cls_values) else -1
            conf = float(conf_values[idx]) if idx < len(conf_values) else 0.0
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            track_id = id_values[idx] if id_values is not None and idx < len(id_values) else idx
            detections.append(self._to_detection(track_id, label, conf, xyxy))

        return detections
