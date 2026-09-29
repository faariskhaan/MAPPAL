"""Create printable ArUco markers, one PNG per zone, with the zone name under it.

Run from the repo root:
    python tools/make_markers.py
Then print the files in markers/ (A4, 100% scale) and tape them at eye level.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

MARKER_PIXELS = 600   # size of the black-and-white marker itself
BORDER = 80           # white quiet zone around the marker (detection needs it)
TEXT_HEIGHT = 140     # space under the marker for the zone name


def make_marker_image(marker_id, zone_name):
    """Return a white image with the marker in the middle and the zone name below."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PIXELS)

    width = MARKER_PIXELS + 2 * BORDER
    height = MARKER_PIXELS + 2 * BORDER + TEXT_HEIGHT
    page = np.full((height, width), 255, dtype=np.uint8)
    page[BORDER:BORDER + MARKER_PIXELS, BORDER:BORDER + MARKER_PIXELS] = marker

    label = f"{zone_name}  (id {marker_id})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 2.0, 4
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    x = (width - text_w) // 2
    y = BORDER + MARKER_PIXELS + BORDER // 2 + text_h
    cv2.putText(page, label, (x, y), font, scale, 0, thickness, cv2.LINE_AA)
    return page


def main(out_dir="markers"):
    os.makedirs(out_dir, exist_ok=True)
    for marker_id, zone_name in config.ZONES.items():
        path = os.path.join(out_dir, f"marker_{marker_id}_{zone_name}.png")
        cv2.imwrite(path, make_marker_image(marker_id, zone_name))
        print("saved", path)


if __name__ == "__main__":
    main()
