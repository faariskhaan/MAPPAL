"""YOLO nano detector: a few snapshots of a zone -> one stable object inventory.

The model is loaded ONCE and only runs on the 2-3 selected snapshots, never on video.
"""

from collections import Counter
from statistics import median_low

import config


def count_objects(class_names, confidences, min_confidence, ignored):
    """Turn one snapshot's detections into {class_name: count}."""
    counts = Counter()
    for name, conf in zip(class_names, confidences):
        if conf >= min_confidence and name not in ignored:
            counts[name] += 1
    return dict(counts)


def combine_snapshots(per_snapshot, min_votes=2):
    """Merge several per-snapshot counts into one stable inventory.

    Stability rule: an object class only counts if it was detected in at least
    `min_votes` snapshots (one-off false detections are dropped). Its count is the
    median of the counts in the snapshots where it was seen (median_low, so the
    result is always a count we actually observed).
    With fewer snapshots than min_votes (e.g. only 1 frame) we need it in all of them.
    """
    min_votes = min(min_votes, len(per_snapshot)) if per_snapshot else min_votes
    seen_in = {}
    for counts in per_snapshot:
        for name, n in counts.items():
            if n > 0:
                seen_in.setdefault(name, []).append(n)
    return {
        name: median_low(values)
        for name, values in sorted(seen_in.items())
        if len(values) >= min_votes
    }


class Detector:
    """Wraps the ultralytics YOLO nano model (CPU only)."""

    def __init__(self, model_path=None, min_confidence=None):
        from ultralytics import YOLO  # imported here so tests never need the model

        self.model = YOLO(model_path or config.YOLO_MODEL)
        self.min_confidence = config.MIN_CONFIDENCE if min_confidence is None else min_confidence
        self.ignored = set(config.IGNORED_CLASSES)

    def detect(self, frame):
        """Run YOLO on ONE snapshot and return {class_name: count}."""
        result = self.model.predict(
            frame, device="cpu", conf=self.min_confidence, imgsz=config.YOLO_IMAGE_SIZE,
            verbose=False,
        )[0]
        names = [result.names[int(c)] for c in result.boxes.cls.tolist()]
        return count_objects(names, result.boxes.conf.tolist(), self.min_confidence, self.ignored)

    def inventory(self, frames):
        """Snapshots -> stable inventory dict, e.g. {"bottle": 2, "backpack": 1}."""
        per_snapshot = [self.detect(frame) for frame in frames]
        return combine_snapshots(per_snapshot, config.MIN_SNAPSHOT_VOTES)
