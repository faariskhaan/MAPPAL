"""MAPPAL v2 - the live app (Week 2: memory + map).

Flow for every camera frame:
  1. detect_zone()        -> which zone marker is visible? (also moves us on the map)
  2. new zone visit?      -> start collecting snapshots (only if the zone was out of
                             view for ZONE_COOLDOWN seconds, so we don't re-scan it
                             every time the marker flickers)
  3. SnapshotCollector    -> keeps the 3 sharpest frames for SNAPSHOT_SECONDS
  4. Detector.inventory() -> YOLO nano on those 3 frames only, in a background
                             thread so the camera window never freezes
  5. process_visit()      -> compare with the last visit (change engine), the normal
                             learner marks each change "routine" or "alert", save the
                             visit + change events in SQLite
  6. TopoMap              -> zone turns orange for an alert, peach for routine, blue
                             if nothing changed; only alerts get the banner
  7. one window: camera on the left, live map on the right, alert banner

Demo Act 3 (--compare): a side panel also shows what a "dumb" pixel-difference
system would report for the same visit (demo/pixel_baseline.py). MAPPAL's own
logic is not changed by this flag.

Run:  python main.py            (webcam)
      python main.py --source http://192.168.1.23:8080/video   (phone)
      python main.py --compare     (adds the pixel-system panel for Act 3)
Keys: q / Esc = quit
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import config
from demo.pixel_baseline import PixelBaseline, verdict
from map.topo_map import TopoMap
from memory import db
from memory.change_engine import compare, describe
from vision.camera import open_camera
from vision.detector import Detector
from vision.snapshot import SnapshotCollector
from zones.aruco_zone import detect_zone, draw_zone

WINDOW = "MAPPAL - The Walking Memory"
BANNER_COLOR = (0, 140, 255)   # orange (BGR)
GREEN, RED = (80, 200, 80), (60, 60, 230)
COMPARE_WIDTH = 330


def should_process(zone, now, last_seen, cooldown):
    """True if this zone should be scanned: never seen before, or out of view long enough."""
    if zone not in last_seen:
        return True
    return now - last_seen[zone] >= cooldown


def process_visit(zone, inventory):
    """Compare with the last visit, then save the visit and its changes.

    Returns (visit_id, changes). The history is read BEFORE saving the new visit,
    otherwise we would compare the visit with itself. The normal learner uses the
    history to mark each change "routine" or "alert".
    """
    history = db.get_visit_history(zone)
    old_inventory = history[-1] if history else None
    changes = compare(zone, old_inventory, inventory, history)
    visit_id = db.save_visit(zone, inventory)
    db.save_events(changes)
    return visit_id, changes


def alert_text(changes, max_items=2):
    """Banner text for a list of changes, e.g. 'SHELF: backpack missing  (+1 more)'."""
    if not changes:
        return ""
    text = " | ".join(describe(c) for c in changes[:max_items])
    if len(changes) > max_items:
        text += f"  (+{len(changes) - max_items} more)"
    return text


def mappal_verdict(changes):
    """What MAPPAL reports for a visit, in a few words (used by the --compare panel)."""
    alerts = sum(1 for c in changes if c["severity"] == "alert")
    if alerts:
        return f"{alerts} ALERT" + ("S" if alerts > 1 else "")
    return "routine change" if changes else "No change"


def draw_compare_panel(width, height, result):
    """Act 3 side panel: the pixel system's verdict next to MAPPAL's for the last visit."""
    img = np.full((height, width, 3), (45, 25, 25), np.uint8)
    cv2.putText(img, "Pixels vs meaning", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    if result is None:
        cv2.putText(img, "Waiting for a zone visit...", (15, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (160, 160, 160), 1, cv2.LINE_AA)
        return img
    pixel_pct, pixel_word = result["pixel"], verdict(result["pixel"])
    pixel_line = "first look" if pixel_pct is None else f"{pixel_pct:.0f}% {pixel_word}"
    mappal_line = result["mappal"]
    rows = [
        (f"Zone: {result['zone']}", (255, 255, 255), 0.7, 80),
        ("Pixel system:", (200, 200, 200), 0.65, 150),
        (pixel_line, RED if pixel_word == "CHANGED" else GREEN, 1.0, 195),
        ("MAPPAL (AI):", (200, 200, 200), 0.65, 280),
        (mappal_line, BANNER_COLOR if "ALERT" in mappal_line else GREEN, 1.0, 325),
    ]
    for text, color, scale, y in rows:
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return img


def draw_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, zone, inventory, status):
    """Status line at the bottom and the object list of the zone on the right."""
    h, w = frame.shape[:2]
    draw_text(frame, status, (20, h - 20), 0.6, (0, 255, 255))
    if zone is None or inventory is None:
        return
    lines = [f"{zone} memory:"] + (
        [f"  {name}: {count}" for name, count in sorted(inventory.items())] or ["  (no objects)"]
    )
    x = max(w - 260, 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, 70), (w - 10, 90 + 28 * len(lines)), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        draw_text(frame, line, (x, 98 + 28 * i), 0.6, (255, 255, 255) if i else (0, 255, 0))


def draw_banner(frame, text):
    """Big orange alert bar near the bottom of the camera view."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 110), (w, h - 50), BANNER_COLOR, -1)
    cv2.putText(frame, "ALERT  " + text, (15, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 2, cv2.LINE_AA)


def fit_height(frame, height):
    """Resize a camera frame to the window height, keeping its shape."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * height / h), height))


def compose(camera_view, map_view, *extra_panels):
    """Camera on the left, map on the right (and any extra panels after it), in one image."""
    return np.hstack([camera_view, map_view, *extra_panels])


def main():
    parser = argparse.ArgumentParser(description="MAPPAL - The Walking Memory")
    parser.add_argument("--source", default=config.CAMERA_SOURCE,
                        help="0 for webcam or a phone stream URL")
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite database file")
    parser.add_argument("--compare", action="store_true",
                        help="Act 3: show the dumb pixel system next to MAPPAL")
    args = parser.parse_args()

    db.init_db(args.db)
    print(f"Loading {config.YOLO_MODEL} on CPU (only once)...")
    detector = Detector()
    cap = open_camera(args.source)

    collector = SnapshotCollector()
    topo = TopoMap()
    worker = ThreadPoolExecutor(max_workers=1)   # runs YOLO off the camera loop
    job = None             # (zone, future, sharpest snapshot) while the AI is analysing
    pixels = PixelBaseline() if args.compare else None
    compare_result = None  # last visit's pixel-vs-MAPPAL result for the side panel
    last_seen = {}         # zone -> last time its marker was visible
    status = "Point the camera at a zone marker"
    banner, banner_until = "", 0.0
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

        # 1. Which zone are we in? Walking into it also moves us on the map.
        zone, corners = detect_zone(frame)
        if zone is not None:
            topo.visit(zone)

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
                job = (visit_zone, worker.submit(detector.inventory, snapshots), snapshots[0])
                status = f"{visit_zone}: AI analysing {len(snapshots)} snapshots..."

        # 5-6. When the AI is done: compare, save, update the map, raise alerts.
        if job is not None and job[1].done():
            visit_zone, future, sharpest = job
            job = None
            try:
                inventory = future.result()
                visit_id, changes = process_visit(visit_zone, inventory)
                topo.set_result(visit_zone, changes)
                alerts = [c for c in changes if c["severity"] == "alert"]
                routine = [c for c in changes if c["severity"] == "routine"]
                if alerts:
                    # Only unusual changes interrupt with a banner.
                    banner = alert_text(alerts)
                    banner_until = now + config.ALERT_SECONDS
                    topo.set_alert(banner)
                    status = f"{visit_zone}: {len(alerts)} alert(s) - visit #{visit_id}"
                elif routine:
                    status = f"{visit_zone}: routine - {alert_text(routine)}"
                else:
                    status = f"{visit_zone}: no change - visit #{visit_id}"
                print(f"[visit #{visit_id}] {visit_zone}: {inventory}  changes: "
                      f"{[describe(c) + ' (' + c['severity'] + ')' for c in changes] or 'none'}")
                if pixels is not None:
                    compare_result = {
                        "zone": visit_zone,
                        "pixel": pixels.compare(visit_zone, sharpest) if sharpest is not None else None,
                        "mappal": mappal_verdict(changes),
                    }
            except Exception as exc:  # keep the demo running if one visit fails
                status = f"{visit_zone}: analysis failed ({exc})"
                print(status)

        # 7. One window: camera + zone memory on the left, live map on the right.
        view = fit_height(frame, config.VIEW_HEIGHT)
        if corners is not None:
            corners = corners * (config.VIEW_HEIGHT / frame.shape[0])
        shown_zone = zone or (job[0] if job else None)
        draw_zone(view, shown_zone, corners)
        draw_panel(view, shown_zone, db.get_last_inventory(shown_zone) if shown_zone else None, status)
        if banner and now < banner_until:
            draw_banner(view, banner)
        panels = [topo.draw(config.MAP_WIDTH, config.VIEW_HEIGHT)]
        if pixels is not None:
            panels.append(draw_compare_panel(COMPARE_WIDTH, config.VIEW_HEIGHT, compare_result))
        cv2.imshow(WINDOW, compose(view, *panels))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    worker.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()
    db.close_db()


if __name__ == "__main__":
    main()
