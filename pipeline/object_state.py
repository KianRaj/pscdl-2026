"""
Tracks each detected object through the persistence state machine:
  DETECTED → (wait P sec) → ACTIVE → (wait C-P sec) → BACKGROUND

P = persistence threshold (seconds before masking)
C = cooldown period (seconds before object absorbed into background)
Active window = [T_intro + P, T_intro + C)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ObjectState(Enum):
    DETECTED = "detected"       # Seen but not yet persistent
    ACTIVE = "active"           # Persistent: should appear in mask
    BACKGROUND = "background"   # Absorbed into background model


@dataclass
class TrackedObject:
    obj_id: int
    t_intro: float                      # Frame timestamp (seconds) when first detected
    bbox: tuple                         # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None   # SAM3 pixel mask (set when ACTIVE)
    state: ObjectState = ObjectState.DETECTED

    def update_state(self, current_time: float, P: float, C: float) -> ObjectState:
        elapsed = current_time - self.t_intro
        if elapsed >= C:
            self.state = ObjectState.BACKGROUND
        elif elapsed >= P:
            self.state = ObjectState.ACTIVE
        else:
            self.state = ObjectState.DETECTED
        return self.state

    @property
    def is_active(self) -> bool:
        return self.state == ObjectState.ACTIVE

    @property
    def is_background(self) -> bool:
        return self.state == ObjectState.BACKGROUND


class ObjectTracker:
    def __init__(self, P: float, C: float, min_area: int = 500, iou_threshold: float = 0.3):
        self.P = P
        self.C = C
        self.min_area = min_area
        self.iou_threshold = iou_threshold
        self._objects: dict[int, TrackedObject] = {}
        self._next_id = 0
        self._background_ids: set[int] = set()

    def update(self, current_time: float, detections: list[tuple]) -> list[TrackedObject]:
        """
        detections: list of (bbox, contour_mask) from change detector
        Returns list of currently ACTIVE objects.
        """
        # Match detections to existing tracked objects
        unmatched = list(range(len(detections)))
        for obj in list(self._objects.values()):
            if obj.is_background:
                continue
            best_iou, best_det = 0.0, -1
            for i in unmatched:
                iou = _bbox_iou(obj.bbox, detections[i][0])
                if iou > best_iou:
                    best_iou, best_det = iou, i
            if best_iou >= self.iou_threshold:
                obj.bbox = detections[best_det][0]
                unmatched.remove(best_det)

        # Register new objects for unmatched detections
        for i in unmatched:
            bbox, _ = detections[i]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_area:
                continue
            obj = TrackedObject(obj_id=self._next_id, t_intro=current_time, bbox=bbox)
            self._objects[self._next_id] = obj
            self._next_id += 1

        # Update states
        for obj in self._objects.values():
            obj.update_state(current_time, self.P, self.C)

        return [o for o in self._objects.values() if o.is_active]

    def get_background_objects(self) -> list[TrackedObject]:
        """Returns objects that just transitioned to BACKGROUND (for bg model update)."""
        new_bg = []
        for obj_id, obj in self._objects.items():
            if obj.is_background and obj_id not in self._background_ids:
                self._background_ids.add(obj_id)
                new_bg.append(obj)
        return new_bg

    def needs_sam3(self, obj_id: int) -> bool:
        """True if this object just became ACTIVE and has no SAM3 mask yet."""
        obj = self._objects.get(obj_id)
        return obj is not None and obj.is_active and obj.mask is None


def _bbox_iou(b1: tuple, b2: tuple) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)
