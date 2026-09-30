"""Pixel baseline tests with synthetic images (no camera)."""

import cv2
import numpy as np

from demo.pixel_baseline import PixelBaseline, percent_changed, to_gray, verdict


def _room():
    """A fake room: dark wall with a few objects-like shapes."""
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(img, (50, 200), (200, 450), (40, 120, 200), -1)
    cv2.circle(img, (400, 300), 60, (200, 200, 60), -1)
    return img


def test_same_picture_is_zero_percent():
    g = to_gray(_room())
    assert percent_changed(g, g) == 0.0


def test_light_change_looks_like_a_big_change_to_pixels():
    """The Act 3 point: nothing moved, but a lamp made the room brighter."""
    brighter = cv2.convertScaleAbs(_room(), alpha=1.0, beta=50)
    pct = percent_changed(to_gray(_room()), to_gray(brighter))
    assert pct > 90
    assert verdict(pct) == "CHANGED"


def test_small_brightness_noise_is_ignored():
    noisy = cv2.convertScaleAbs(_room(), alpha=1.0, beta=10)   # below threshold 30
    assert percent_changed(to_gray(_room()), to_gray(noisy)) == 0.0


def test_baseline_remembers_each_zone_separately():
    pb = PixelBaseline()
    assert pb.compare("Shelf", _room()) is None          # first time: nothing to compare
    assert pb.compare("Desk", np.zeros((480, 640, 3), np.uint8)) is None
    assert pb.compare("Shelf", _room()) == 0.0
    pb.reset()
    assert pb.compare("Shelf", _room()) is None


def test_different_frame_sizes_can_be_compared():
    small = cv2.resize(_room(), (320, 240))
    assert percent_changed(to_gray(_room()), to_gray(small)) < 5


def test_verdict():
    assert verdict(None) == "learning"
    assert verdict(2.0) == "no change"
    assert verdict(68.0) == "CHANGED"
