# MAPPAL — The Walking Memory

Final Year Project. Software prototype only. **No hardware.**

## What this project is
One camera is carried through a building. MAPPAL:
1. Recognises which zone it is in (ArUco marker on the wall).
2. Uses AI (YOLO nano, pretrained on COCO) to turn 2–3 snapshots of that zone into an object inventory.
3. Compares the inventory with the last visit, reports what changed, learns what is "normal", and draws a live topological map (zones = nodes, walk order = edges).

Pitch line: *Cameras see pixels. MAPPAL's AI sees meaning.*
Core trick: *AI on moments, not on video* — run the detector on a few snapshots per zone, never on every frame.

## Scope — LOCKED (3 features)
1. Zone recognition (ArUco)
2. AI zone memory (YOLO nano → inventory → SQLite)
3. AI change intelligence + live map (compare, learn normal, draw map)

Do NOT add features outside this list. If a task seems to need one, stop and ask.
Out of scope for now: SLAM, depth, robots, VIO, servo motors, cloud APIs, LLMs, face recognition, web dashboards, marker-free place recognition.

## Tech stack — LOCKED
Python 3.10+, `opencv-contrib-python` (camera, ArUco, drawing the map), `ultralytics` (YOLO nano), `numpy`, `sqlite3` (standard library), `pytest`.
Do not add any other dependency without asking. The map is drawn with OpenCV, not Matplotlib/Pygame.

## Hardware limits
Team laptops are ordinary (no GPU). Everything must run on CPU.
Never run YOLO on every frame. Only on the selected snapshots.

## Folder structure
```
mappal/
  CLAUDE.md
  requirements.txt
  config.py              # camera source, zone names, thresholds
  main.py                # the live app
  tools/make_markers.py  # print ArUco markers
  zones/aruco_zone.py    # marker id -> zone name
  vision/snapshot.py     # pick 2-3 sharp frames per zone visit
  vision/detector.py     # YOLO nano -> inventory dict
  memory/db.py           # SQLite: zones, visits, inventories, events
  memory/change_engine.py
  memory/normal_learner.py
  map/topo_map.py        # graph + OpenCV drawing
  demo/pixel_baseline.py # "dumb" pixel-diff system for Act 3 of the demo
  tests/
```

## Data shapes (keep these stable)
- Inventory: `dict[str, int]`, e.g. `{"bottle": 2, "backpack": 1}` (COCO class names).
- Change: `{"zone": str, "object": str, "type": "missing"|"new"|"count_changed", "before": int, "after": int, "severity": "routine"|"alert"}`
- Zone colour on map: blue = no change, orange = changed, grey = not visited for `STALE_SECONDS`.

## Camera
`config.CAMERA_SOURCE` is either `0` (laptop webcam) or a phone stream URL (IP Webcam / DroidCam, e.g. `http://192.168.x.x:8080/video`).

## Rules for every task
- Keep code simple and readable; a student must be able to explain it to a supervisor.
- After writing code, explain in plain words what each function does and why.
- Write or update a pytest test for any logic (change engine, learner, db). Tests must not need a camera or the YOLO model — use fake inventories.
- Stay inside the file(s) named in the task unless you ask first.
- Never commit large files (models, videos, the .db file). Keep them in .gitignore.
