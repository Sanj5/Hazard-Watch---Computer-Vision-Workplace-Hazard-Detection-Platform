from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

from src.utils.types import Detection


class TrajectoryBuffer:
    def __init__(self, maxlen: int = 30):
        self.maxlen = maxlen
        self._tracks: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=maxlen))

    @staticmethod
    def _center(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections: List[Detection]) -> None:
        for d in detections:
            self._tracks[d.track_id].append(self._center(d.bbox))

    def get_history(self, track_id: int) -> List[Tuple[float, float]]:
        return list(self._tracks.get(track_id, []))
