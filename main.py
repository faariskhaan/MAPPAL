"""MAPPAL v1 - the live app (Week 1).

Flow for every camera frame:
  1. detect_zone()        -> which zone marker is visible?
  2. new zone visit?      -> start collecting snapshots (only if the zone was out of
                             view for ZONE_COOLDOWN seconds, so we don't re-scan it
                             every time the marker flickers)
  3. SnapshotCollector    -> keeps the 3 sharpest frames for SNAPSHOT_SECONDS
  4. Detector.inventory() -> YOLO nano on those 3 frames only, in a background
                             thread so the camera window never freezes
  5. db.save_visit()      -> the inventory is stored in SQLite
  6. screen shows zone name + object list

Run:  python main.py            (webcam)
      python main.py --source http://192.168.1.23:8080/video   (phone)
Keys: q / Esc = quit
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

import config
from memory import db
from vision.camera import open_camera
from vision.detector import Detector
from vision.snapshot import SnapshotCollector
from zones.aruco_zone import detect_zone, draw_zone

WINDOW = "MAPPAL - The Walking Memory"


def should_process(zone, now, last_seen, cooldown):
    """True if this zone should be scanned: never seen before, or out of view long enough."""
    if zone not in last_seen:
        return True
    return now - last_seen[zone] >= cooldown


def draw_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, zone, inventory, status):
    """Status line at the bottom and the object list of the zone on the right."""
    h, w = frame.shape[:2]
    draw_text(frame, status, (20, h - 20), 0.7, (0, 255, 255))
    if zone is None or inventory is None:
        return
    lines = [f"{zone} memory:"] + (
        [f"  {name}: {count}" for name, count in sorted(inventory.items())] or ["  (no objects)"]
    )
    x = max(w - 300, 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, 70), (w - 10, 90 + 30 * len(lines)), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        draw_text(frame, line, (x, 100 + 30 * i), 0.7, (255, 255, 255) if i else (0, 255, 0))


def main():
    parser = argparse.ArgumentParser(description="MAPPAL - The Walking Memory")
    parser.add_argument("--source", default=config.CAMERA_SOURCE,
                        help="0 for webcam or a phone stream URL")
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite database file")
    args = parser.parse_args()

    db.init_db(args.db)
    print(f"Loading {config.YOLO_MODEL} on CPU (only once)...")
    detector = Detector()
    cap = open_camera(args.source)

    collector = SnapshotCollector()
    worker = ThreadPoolExecutor(max_workers=1)   # runs YOLO off the camera loop
    job = None             # (zone, future) while the AI is analysing
    last_seen = {}         # zone -> last time its marker was visible
    status = "Point the camera at a zone marker"
    failures = 0

    print("Running. Press q or Esc in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 50:
                print("Camera stream lost - stopping.")
                break
            continue
        failures = 0
        now = time.monotonic()

        # 1. Which zone are we in?
        zone, corners = detect_zone(frame)

        # 2. Start a new visit when a zone (re)appears after the cooldown.
        if zone is not None:
            busy_with_zone = (collector.zone == zone) or (job is not None and job[0] == zone)
            if not busy_with_zone and should_process(zone, now, last_seen, config.ZONE_COOLDOWN):
                if collector.active:
                    # Walked into another zone before finishing: forget the old one
                    # so it is scanned again next time.
                    last_seen.pop(collector.zone, None)
                collector.start(zone)
                status = f"{zone}: collecting snapshots..."
            last_seen[zone] = now

        # 3. Feed frames to the snapshot picker (cheap, no AI here).
        if collector.active:
            status = f"{collector.zone}: collecting snapshots {collector.elapsed():.1f}s"
            if collector.add(frame) and job is None:
                visit_zone, snapshots = collector.finish()
                # 4. AI on moments: YOLO runs on the 3 sharp snapshots, in the background.
                job = (visit_zone, worker.submit(detector.inventory, snapshots))
                status = f"{visit_zone}: AI analysing {len(snapshots)} snapshots..."

        # 5. When the AI is done, save the visit in the database.
        if job is not None and job[1].done():
            visit_zone, future = job
            job = None
            try:
                inventory = future.result()
                visit_id = db.save_visit(visit_zone, inventory)
                status = f"{visit_zone}: saved visit #{visit_id} to memory"
                print(f"[visit #{visit_id}] {visit_zone}: {inventory}")
            except Exception as exc:  # keep the demo running if one visit fails
                status = f"{visit_zone}: analysis failed ({exc})"
                print(status)

        # 6. Show zone name + what MAPPAL remembers about it.
        shown_zone = zone or (job[0] if job else None)
        draw_zone(frame, shown_zone, corners)
        draw_panel(frame, shown_zone, db.get_last_inventory(shown_zone) if shown_zone else None, status)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    worker.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()
    db.close_db()


if __name__ == "__main__":
    main()
