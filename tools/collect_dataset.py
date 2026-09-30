"""Collect our own photos of the zones for training the zone classifier.

Run from the repo root:
    python tools/collect_dataset.py walk1
    python tools/collect_dataset.py walk2 --source http://192.168.1.23:8080/video

Keys:
    0-3    select a zone (Desk, Shelf, Door, Window - config.ZONES in id order)
    4      select "Other" (walls, floor, hallway, blur, hand over the lens)
    SPACE  start / pause recording
    q/Esc  quit
While recording, one frame is saved every SAVE_INTERVAL seconds to
dataset/<session>/<class>/<class>_<timestamp>.jpg (JPEG quality 90, longest side 640).
Use a separate session for training (walk1), validation (walk2) and test (test).
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from vision.camera import open_camera  # noqa: E402

MAX_SIDE = 640
JPEG_QUALITY = 90


def class_names():
    """Zones in id order, then the extra class: ["Desk", "Shelf", "Door", "Window", "Other"]."""
    return [config.ZONES[i] for i in sorted(config.ZONES)] + [config.EXTRA_CLASS]


def class_for_key(key):
    """Keyboard key code -> class name, or None if the key does not select a class."""
    names = class_names()
    if ord("0") <= key < ord("0") + len(names):
        return names[key - ord("0")]
    return None


def resize_longest(frame, max_side=MAX_SIDE):
    """Shrink so the longest side is at most max_side (never enlarges)."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(".jpg"))


def save_frame(frame, folder, class_name):
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(folder, f"{class_name}_{stamp}.jpg")
    cv2.imwrite(path, resize_longest(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return path


def draw_overlay(frame, session, selected, recording, counts):
    """Selected class, RECORDING/PAUSED, image count per class, and the reminder."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
    state, color = ("RECORDING", (0, 0, 255)) if recording else ("PAUSED", (0, 200, 255))
    cv2.putText(frame, f"Session: {session}   Class: {selected}   {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (w - 25, 22), 10, (0, 0, 255), -1)
    y = 75
    for i, name in enumerate(class_names()):
        mark = ">" if name == selected else " "
        text = f"{mark} [{i}] {name}: {counts[name]}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if name == selected else (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    tip = "move the camera slowly, change angle and distance  |  SPACE rec/pause  q quit"
    cv2.putText(frame, tip, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, tip, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Collect zone photos for training")
    parser.add_argument("session", help="session name, e.g. walk1, walk2, test")
    parser.add_argument("--source", default=config.CAMERA_SOURCE,
                        help="0 for webcam or a phone stream URL")
    args = parser.parse_args()

    session_dir = os.path.join(config.DATASET_DIR, args.session)
    counts = {name: count_images(os.path.join(session_dir, name)) for name in class_names()}
    selected = class_names()[0]
    recording = False
    last_save = 0.0

    cap = open_camera(args.source)
    print(f"Saving into {session_dir}/<class>/ . Keys: 0-4 class, SPACE record, q quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera - stopping.")
            break
        now = time.monotonic()
        if recording and now - last_save >= config.SAVE_INTERVAL:
            save_frame(frame, os.path.join(session_dir, selected), selected)
            counts[selected] += 1
            last_save = now

        view = frame.copy()      # draw on a copy so the saved photos stay clean
        draw_overlay(view, args.session, selected, recording, counts)
        cv2.imshow("MAPPAL - dataset collector", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            recording = not recording
        new_class = class_for_key(key)
        if new_class:
            selected = new_class
            recording = False    # pause when switching, so no photo lands in the wrong class

    cap.release()
    cv2.destroyAllWindows()
    print("Images in this session:", counts)


if __name__ == "__main__":
    main()
