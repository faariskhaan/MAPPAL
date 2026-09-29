"""Zone recognition: find an ArUco marker in a camera frame and return its zone name."""

import cv2
import numpy as np

import config

# Build the detector once (not on every frame).
_dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
_detector = cv2.aruco.ArucoDetector(_dictionary, cv2.aruco.DetectorParameters())


def detect_zone(frame):
    """Return (zone_name, corners) for the biggest known marker in the frame.

    zone_name is None when no known marker is visible.
    corners is a (4, 2) float array of the marker's corner pixels, or None.
    If several markers are visible we pick the biggest one = the closest zone.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    all_corners, ids, _rejected = _detector.detectMarkers(gray)
    if ids is None:
        return None, None

    best_zone, best_corners, best_area = None, None, 0.0
    for marker_corners, marker_id in zip(all_corners, ids.flatten()):
        zone = config.ZONES.get(int(marker_id))
        if zone is None:
            continue  # a marker that is not one of our zones
        pts = marker_corners.reshape(4, 2)
        area = cv2.contourArea(pts.astype(np.float32))
        if area > best_area:
            best_zone, best_corners, best_area = zone, pts, area
    return best_zone, best_corners


def draw_zone(frame, zone_name, corners):
    """Draw the marker outline and the zone name on the frame (in place)."""
    if corners is not None:
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    text = f"Zone: {zone_name}" if zone_name else "Zone: -"
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    return frame
