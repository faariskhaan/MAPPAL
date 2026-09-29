"""Zone recognition tests using generated marker images (no camera needed)."""

import cv2
import numpy as np

import config
from tools.make_markers import make_marker_image
from vision.camera import parse_source
from zones.aruco_zone import detect_zone


def _scene_with_marker(marker_id, size=640):
    """A grey 'wall' with a printed marker page pasted on it, as a BGR image."""
    wall = np.full((size, size), 128, dtype=np.uint8)
    page = cv2.resize(make_marker_image(marker_id, config.ZONES[marker_id]), (200, 240))
    wall[100:340, 150:350] = page
    return cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)


def test_every_zone_marker_is_recognised():
    for marker_id, zone_name in config.ZONES.items():
        zone, corners = detect_zone(_scene_with_marker(marker_id))
        assert zone == zone_name
        assert corners.shape == (4, 2)


def test_no_marker_means_no_zone():
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert detect_zone(blank) == (None, None)


def test_unknown_marker_id_is_ignored():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    wall = np.full((480, 640), 255, dtype=np.uint8)
    wall[100:300, 100:300] = cv2.aruco.generateImageMarker(dictionary, 42, 200)
    zone, _ = detect_zone(cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR))
    assert zone is None


def test_camera_source_parsing():
    assert parse_source("0") == 0
    assert parse_source(1) == 1
    assert parse_source("http://192.168.1.5:8080/video") == "http://192.168.1.5:8080/video"
