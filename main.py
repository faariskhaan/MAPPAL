"""MAPPAL v2 - the live app (Week 2: memory + map).

Flow for every camera frame:
  1. which zone are we in?
     - our trained zone classifier (MobileNetV2) looks at every CLASSIFY_EVERY-th
       frame; the ZoneSmoother confirms a zone after ZONE_CONSECUTIVE sure guesses
       -> source "AI"
     - otherwise the ArUco marker decides -> source "marker" (the fallback)
     (walking into a zone also moves us on the map)
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
      python main.py --demo --compare   (supervisor demo: full screen, large text)
      python main.py --marker-only      (ignore the trained zone model, ArUco only)
Keys: q / Esc = quit,  r = reset memory (demo mode)
"""

import argparse
import os
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
from vision.zone_classifier import ZoneClassifier, ZoneSmoother
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


def choose_zone(ai_zone, marker_zone):
    """Zone decision: a zone confirmed by our AI wins, otherwise the marker (fallback).

    Returns (zone or None, source) with source "AI", "marker" or None.
    """
    if ai_zone is not None:
        return ai_zone, "AI"
    if marker_zone is not None:
        return marker_zone, "marker"
    return None, None


def load_zone_classifier(path, disabled):
    """Load our trained zone model, or return None (then the app uses markers only)."""
    if disabled:
        print("Zone classifier off (--marker-only): using ArUco markers only.")
        return None
    if not os.path.exists(path):
        print(f"No trained zone model at '{path}' - using ArUco markers only.\n"
              "  Train one with training/train_zone.py and put it there to enable the AI.")
        return None
    try:
        clf = ZoneClassifier(path)
    except Exception as exc:  # a broken file must not stop the demo
        print(f"Could not load zone model '{path}' ({exc}) - using ArUco markers only.")
        return None
    print(f"Zone classifier loaded: {path} classes {clf.class_names}")
    return clf


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


def zone_label(zone, source, ai_guess):
    """'Shelf (AI 0.94)', 'Shelf (marker)' or just the zone name."""
    if zone is None:
        return None
    if source == "AI" and ai_guess is not None:
        return f"{zone} (AI {ai_guess[1]:.2f})"
    if source == "marker":
        return f"{zone} (marker)"
    return zone


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


def draw_panel(frame, zone, inventory, status, scale=1.0):
    """Status line at the bottom and the object list of the zone on the right."""
    h, w = frame.shape[:2]
    draw_text(frame, status, (20, h - int(20 * scale)), 0.6 * scale, (0, 255, 255))
    if zone is None or inventory is None:
        return
    lines = [f"{zone} memory:"] + (
        [f"  {name}: {count}" for name, count in sorted(inventory.items())] or ["  (no objects)"]
    )
    line_h = int(28 * scale)
    x = max(w - int(260 * scale), 10)
    top = int(105 * scale)   # below the zone name and the AI guess line
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, top), (w - 10, top + 20 + line_h * len(lines)), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        draw_text(frame, line, (x, top + line_h + line_h * i), 0.6 * scale,
                  (255, 255, 255) if i else (0, 255, 0))


def draw_banner(frame, text, scale=1.0):
    """Big orange alert bar near the bottom of the camera view."""
    h, w = frame.shape[:2]
    bottom = h - int(50 * scale)
    cv2.rectangle(frame, (0, bottom - int(60 * scale)), (w, bottom), BANNER_COLOR, -1)
    text = "ALERT  " + text
    font_scale = 0.8 * scale
    while font_scale > 0.4:   # shrink long alerts so they fit on a narrow camera view
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        if tw <= w - 30:
            break
        font_scale -= 0.05
    cv2.putText(frame, text, (15, bottom - int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 2, cv2.LINE_AA)


def draw_title_bar(width, scale=1.0):
    """Demo mode title strip across the top of the window."""
    height = int(50 * scale)
    bar = np.full((height, width, 3), (20, 20, 20), np.uint8)
    # OpenCV fonts are plain ASCII, so the title uses "-" instead of an em dash.
    cv2.putText(bar, "MAPPAL - The Walking Memory", (15, int(35 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    keys = "[R] reset memory   [Q] quit"
    (tw, _), _ = cv2.getTextSize(keys, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 1)
    cv2.putText(bar, keys, (width - tw - 15, int(32 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale, (170, 170, 170), 1, cv2.LINE_AA)
    return bar


def render_scaled(draw, width, height, scale):
    """Draw a panel smaller and blow it up, so all its text and boxes get `scale` times bigger."""
    if scale == 1.0:
        return draw(width, height)
    small = draw(int(width / scale), int(height / scale))
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


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
    parser.add_argument("--demo", action="store_true",
                        help="supervisor demo: full screen, large text, title bar, R = reset memory")
    parser.add_argument("--marker-only", "--no-model", dest="marker_only", action="store_true",
                        help="do not use the trained zone classifier, ArUco markers only")
    parser.add_argument("--zone-model", default=config.ZONE_MODEL_PATH,
                        help="trained zone classifier file")
    args = parser.parse_args()
    ui = config.DEMO_UI_SCALE if args.demo else 1.0

    db.init_db(args.db)
    print(f"Loading {config.YOLO_MODEL} on CPU (only once)...")
    detector = Detector()
    zone_clf = load_zone_classifier(args.zone_model, args.marker_only)
    smoother = ZoneSmoother()
    ai_guess = None        # (label, confidence) of the classifier's latest look
    ai_zone = None         # zone confirmed by the smoother, or None
    frame_no = 0
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

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)      # resizable, keeps the picture's shape
    if args.demo:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Running. Press q or Esc in the window to quit" + (", r to reset memory." if args.demo else "."))
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

        # 1. Which zone are we in? Our AI first (every CLASSIFY_EVERY frames), marker as fallback.
        frame_no += 1
        if zone_clf is not None and frame_no % config.CLASSIFY_EVERY == 0:
            ai_guess = zone_clf.predict(frame)
            ai_zone = smoother.update(*ai_guess)
        marker_zone, corners = detect_zone(frame)
        zone, source = choose_zone(ai_zone, marker_zone)
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
        draw_zone(view, zone_label(shown_zone, source if zone else None, ai_guess), corners)
        if zone_clf is not None and ai_guess is not None:
            draw_text(view, f"AI guess: {ai_guess[0]} {ai_guess[1]:.2f}", (20, int(85 * ui)),
                      0.6 * ui, (255, 200, 120))
        draw_panel(view, shown_zone, db.get_last_inventory(shown_zone) if shown_zone else None,
                   status, ui)
        if banner and now < banner_until:
            draw_banner(view, banner, ui)
        panels = [render_scaled(topo.draw, config.MAP_WIDTH, config.VIEW_HEIGHT, ui)]
        if pixels is not None:
            panels.append(render_scaled(
                lambda w, h: draw_compare_panel(w, h, compare_result),
                COMPARE_WIDTH, config.VIEW_HEIGHT, ui))
        window_img = compose(view, *panels)
        if args.demo:
            window_img = np.vstack([draw_title_bar(window_img.shape[1], ui), window_img])
        cv2.imshow(WINDOW, window_img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if args.demo and key in (ord("r"), ord("R")):
            # Reset memory: empty database, empty map, forget pixel snapshots.
            db.reset_db()
            topo = TopoMap()
            if pixels is not None:
                pixels.reset()
            collector.cancel()
            job = None            # a running analysis is ignored
            last_seen.clear()
            banner, compare_result = "", None
            status = "Memory reset - walk the zones to teach MAPPAL again"
            print(status)

    worker.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()
    db.close_db()


if __name__ == "__main__":
    main()
