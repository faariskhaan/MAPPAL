"""Live test of zone recognition: shows the camera and the zone name of any visible marker.

Run from the repo root:
    python tools/zone_viewer.py                                   # laptop webcam
    python tools/zone_viewer.py --source http://192.168.1.23:8080/video   # phone
Press q or Esc to quit.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from vision.camera import open_camera  # noqa: E402
from zones.aruco_zone import detect_zone, draw_zone  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="MAPPAL zone viewer")
    parser.add_argument("--source", default=config.CAMERA_SOURCE,
                        help="0 for webcam or a phone stream URL")
    args = parser.parse_args()

    cap = open_camera(args.source)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera - stopping.")
            break
        zone, corners = detect_zone(frame)
        draw_zone(frame, zone, corners)
        cv2.imshow("MAPPAL - zone viewer", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
