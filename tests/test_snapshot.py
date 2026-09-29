"""Snapshot picker tests with synthetic images and a fake clock (no camera)."""

import cv2
import numpy as np

from vision.snapshot import SnapshotCollector, sharpness


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _checkerboard(blur=0):
    """Sharp pattern; blur > 0 makes it softer (lower sharpness)."""
    tile = np.kron([[0, 255] * 8, [255, 0] * 8] * 6, np.ones((20, 20))).astype(np.uint8)
    img = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def test_blurry_frame_scores_lower():
    assert sharpness(_checkerboard(blur=5)) < sharpness(_checkerboard())


def test_keeps_the_three_sharpest_frames_in_order():
    clock = FakeClock()
    c = SnapshotCollector(seconds=2, count=3, clock=clock)
    c.start("Shelf")
    blurs = [6, 0, 4, 1, 8, 2]          # 0 = sharpest
    for b in blurs:
        c.add(_checkerboard(blur=b))
    clock.now = 2.0
    assert c.done()
    zone, frames = c.finish()
    assert zone == "Shelf"
    scores = [sharpness(f) for f in frames]
    expected = sorted((sharpness(_checkerboard(blur=b)) for b in blurs), reverse=True)[:3]
    assert np.allclose(scores, expected)


def test_not_done_until_time_is_up():
    clock = FakeClock()
    c = SnapshotCollector(seconds=2, count=3, clock=clock)
    c.start("Desk")
    clock.now = 1.0
    assert c.add(_checkerboard()) is False
    clock.now = 2.5
    assert c.add(_checkerboard()) is True


def test_inactive_collector_ignores_frames_and_resets_after_finish():
    c = SnapshotCollector(seconds=0, count=3, clock=FakeClock())
    assert c.add(_checkerboard()) is False
    c.start("Door")
    c.add(_checkerboard())
    c.finish()
    assert not c.active


def test_stored_frames_are_copies():
    c = SnapshotCollector(seconds=0, count=1, clock=FakeClock())
    c.start("Window")
    frame = _checkerboard()
    c.add(frame)
    frame[:] = 0                         # camera buffer gets overwritten
    _zone, frames = c.finish()
    assert frames[0].any()
