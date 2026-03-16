from typing import List, Dict

from src.utils.types import Detection, PPEViolation


class PPEDetector:
    def __init__(self, required_items=("helmet", "vest")):
        self.required_items = set(required_items)

    @staticmethod
    def _is_inside(person_box, item_box) -> bool:
        px1, py1, px2, py2 = person_box
        ix1, iy1, ix2, iy2 = item_box
        cx = (ix1 + ix2) / 2
        cy = (iy1 + iy2) / 2
        return px1 <= cx <= px2 and py1 <= cy <= py2

    def detect_violations(self, detections: List[Detection]) -> List[PPEViolation]:
        persons = [d for d in detections if d.label == "person"]
        ppe_items = [d for d in detections if d.label in self.required_items]

        violations: List[PPEViolation] = []
        for person in persons:
            present = set()
            for item in ppe_items:
                if self._is_inside(person.bbox, item.bbox):
                    present.add(item.label)
            missing = sorted(self.required_items - present)
            if missing:
                conf = max(0.4, min(1.0, person.confidence))
                violations.append(
                    PPEViolation(
                        person_track_id=person.track_id,
                        missing_items=missing,
                        confidence=conf,
                    )
                )
        return violations
