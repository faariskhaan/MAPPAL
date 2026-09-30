"""The "dumb" pixel system for Act 3 of the demo.

It does what a normal security camera does: compare the new picture of a zone with the
last one, pixel by pixel. It has no idea what an object is. Turn on a lamp or close a
curtain and almost every pixel gets brighter or darker -> it screams "CHANGED" while
nothing in the room actually moved. MAPPAL's AI compares objects, not pixels, so it
correctly says "No change".

This module is only used by `python main.py --compare`. It never changes MAPPAL's logic.
"""

import cv2
import numpy as np

import config

COMPARE_SIZE = (320, 240)   # both snapshots are shrunk to this size before comparing


def to_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, COMPARE_SIZE)


def percent_changed(old_gray, new_gray, threshold=None):
    """Percentage of pixels whose grayscale difference is above the threshold (default 30)."""
    threshold = config.PIXEL_THRESHOLD if threshold is None else threshold
    diff = cv2.absdiff(old_gray, new_gray)
    return 100.0 * np.count_nonzero(diff > threshold) / diff.size


class PixelBaseline:
    """Remembers the last sharpest snapshot of every zone (in memory only)."""

    def __init__(self):
        self.last = {}   # zone -> grayscale snapshot

    def compare(self, zone, frame):
        """Store this snapshot and return the % of changed pixels vs the last one (None the first time)."""
        gray = to_gray(frame)
        old = self.last.get(zone)
        self.last[zone] = gray
        return None if old is None else percent_changed(old, gray)

    def reset(self):
        self.last = {}


def verdict(percent):
    """What the pixel system would report on screen."""
    if percent is None:
        return "learning"
    return "CHANGED" if percent >= config.PIXEL_CHANGED_PERCENT else "no change"
