"""Pick the sharpest few frames of a zone visit ("AI on moments, not on video").

Non-blocking design: the main loop keeps reading the camera and hands each frame
to SnapshotCollector.add(). The collector remembers only the N sharpest frames and
says when SNAPSHOT_SECONDS have passed. Nothing here waits or sleeps, so the camera
window never freezes.
"""

import time

import cv2

import config


def sharpness(frame):
    """Variance of the Laplacian: high = crisp edges, low = blurry (motion blur, out of focus)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class SnapshotCollector:
    """Collects frames for a fixed time and keeps the sharpest ones."""

    def __init__(self, seconds=None, count=None, clock=time.monotonic):
        self.seconds = config.SNAPSHOT_SECONDS if seconds is None else seconds
        self.count = config.SNAPSHOT_COUNT if count is None else count
        self.clock = clock          # injectable so tests can fake time
        self.zone = None
        self.start_time = None
        self._best = []             # list of (score, frame), sharpest first

    @property
    def active(self):
        return self.zone is not None

    def start(self, zone):
        """Begin a new collection for this zone (forgets any previous frames)."""
        self.zone = zone
        self.start_time = self.clock()
        self._best = []

    def cancel(self):
        self.zone = None
        self.start_time = None
        self._best = []

    def elapsed(self):
        return 0.0 if self.start_time is None else self.clock() - self.start_time

    def add(self, frame):
        """Offer one frame. Returns True when the collection time is over."""
        if not self.active:
            return False
        score = sharpness(frame)
        if len(self._best) < self.count or score > self._best[-1][0]:
            self._best.append((score, frame.copy()))
            self._best.sort(key=lambda item: item[0], reverse=True)
            del self._best[self.count:]
        return self.done()

    def done(self):
        return self.active and self.elapsed() >= self.seconds

    def finish(self):
        """Return (zone, sharpest frames) and reset the collector."""
        zone = self.zone
        frames = [frame for _score, frame in self._best]
        self.cancel()
        return zone, frames
