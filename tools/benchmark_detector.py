"""Speed test: how many seconds does YOLO nano need per snapshot on THIS laptop (CPU)?

Run from the repo root:
    python tools/benchmark_detector.py                  # uses 3 sample images
    python tools/benchmark_detector.py a.jpg b.jpg c.jpg
Target: under ~1-2 s per snapshot = "runs on a normal laptop" is proven.
"""

import os
import platform
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from vision.detector import Detector  # noqa: E402

RUNS = 10


def sample_images():
    """Three sample photos: the two that ship with ultralytics + a mirrored copy."""
    from ultralytics.utils import ASSETS

    bus = cv2.imread(str(ASSETS / "bus.jpg"))
    zidane = cv2.imread(str(ASSETS / "zidane.jpg"))
    return [bus, zidane, cv2.flip(bus, 1)]


def main():
    paths = sys.argv[1:]
    images = [cv2.imread(p) for p in paths] if paths else sample_images()
    if any(img is None for img in images):
        sys.exit("Could not read one of the images.")

    print(f"Loading {config.YOLO_MODEL} (CPU)...")
    t0 = time.perf_counter()
    detector = Detector()
    print(f"Model load: {time.perf_counter() - t0:.2f} s (happens once at start-up)")

    detector.inventory(images)  # warm-up run, not timed

    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        inv = detector.inventory(images)
        times.append(time.perf_counter() - t0)
        print(f"run {i + 1:2d}: {times[-1]:.2f} s for {len(images)} snapshots  {inv}")

    per_visit = sum(times) / len(times)
    per_snapshot = per_visit / len(images)
    print("-" * 60)
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Average per zone visit ({len(images)} snapshots): {per_visit:.2f} s")
    print(f"Average per snapshot: {per_snapshot:.3f} s")
    print("PASS - fast enough for a normal laptop" if per_snapshot < 2.0
          else "SLOW - consider YOLO_IMAGE_SIZE = 320 in config.py")


if __name__ == "__main__":
    main()
