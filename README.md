# MAPPAL - The Walking Memory
Final Year Project. Software prototype only. No hardware.

*Cameras see pixels. MAPPAL's AI sees meaning.*

One camera is carried through a building. MAPPAL:
1. **Recognises the zone** it is in from an ArUco marker on the wall.
2. **Remembers what is there:** YOLO nano (AI, CPU only) turns 3 sharp snapshots of the zone into an object list, saved in SQLite. *AI on moments, not on video.*
3. **Notices what changed:** compares with the last visit, learns what normally changes (routine) vs what is unusual (alert), and draws a live map of the zones.

## 1. Install (Windows, CPU only, Python 3.10+)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
The YOLO model (`yolov8n.pt`, ~6 MB) downloads automatically the first time.
Check everything works:
```bash
python -m pytest
python tools/benchmark_detector.py
```
The benchmark prints seconds per snapshot (ours: **0.075 s** on an ordinary Intel laptop CPU).

## 2. Print the markers
```bash
python tools/make_markers.py
```
Print the 4 PNGs in `markers/` at 100% scale (Desk, Shelf, Door, Window). Tape each one at eye level next to its zone, in good light, flat (not curled).
Test them live with `python tools/zone_viewer.py`.

## 3. Use a phone as the camera
1. Install **IP Webcam** (Android) or **DroidCam** on the phone.
2. Phone and laptop on the **same Wi-Fi**. Start the server in the app and note the URL.
3. Run with that URL:
```bash
python main.py --source http://192.168.1.23:8080/video
```
(IP Webcam uses port 8080, DroidCam 4747.) Without `--source` the laptop webcam is used; you can also set `CAMERA_SOURCE` in `config.py`.

## 4. Run modes
| Command | What you get |
|---|---|
| `python main.py` | Camera + zone memory on the left, live map on the right |
| `python main.py --compare` | Adds the "Pixels vs meaning" panel (Act 3) |
| `python main.py --demo --compare` | Supervisor demo: full screen, large text, title bar |
| `python main.py --marker-only` | Ignore our trained zone model, ArUco markers only |

Keys: `q`/`Esc` quit, `r` reset memory (demo mode). Deleting `mappal.db` also resets the memory.

Map colours: **blue** = no change, **peach** = routine change, **orange** = alert, **grey** = not visited for 2 minutes. The zone you are in has a white outline.

## 5. The 3-act demo script ("The Trick")
**Before the supervisor arrives:** press `r` for a fresh memory, then do 3 "learning" walks through all zones. On each walk, alternately take the cup away from the Desk and put it back (so MAPPAL learns the cup comes and goes there). Leave the backpack on the Shelf every time.

Start: `python main.py --demo --compare --source <phone url>`

**Act 1 - The walk (MAPPAL remembers).**
Walk Desk → Shelf → Door → Window, ~2 s at each marker.
Say: *"It recognises each zone and uses AI on just 3 snapshots, not on video, to list what is there. The map on the right builds as I walk: places and how they connect."*

**Act 2 - The change (MAPPAL notices, and knows what matters).**
Someone quietly removes the backpack from the Shelf and takes the cup away from the Desk (or puts it back). Walk again (stay away from each zone ≥ 5 s before returning).
- Desk: cup changed → **peach, no banner**. *"The cup moves all the time here. MAPPAL learned that is normal."*
- Shelf: **`ALERT SHELF: backpack missing`**, zone turns orange. *"The backpack has always been here. That is unusual, so it alerts."*

**Act 3 - The trick (pixels vs meaning).**
Change the light at the Shelf (turn on a lamp or open a curtain; **never full darkness**). Visit the Shelf again.
- Pixel system: **~70-95% CHANGED**. MAPPAL: **No change**.
- Say: *"A normal camera compares pixels, so a lamp looks like a burglary. MAPPAL compares objects. Cameras see pixels, MAPPAL's AI sees meaning."*

## 6. Demo day checklist
- [ ] Markers printed and taped at eye level, good light
- [ ] Objects are COCO classes: backpack, bottle, cup, laptop, book, cell phone, chair
- [ ] Fresh memory (`r`), 3 learning walks done before the supervisor arrives
- [ ] Act 3: change light with a lamp or curtain, never full darkness
- [ ] Laptop plugged in, phone and laptop on the same Wi-Fi
- [ ] Backup: laptop webcam (`python main.py --demo --compare`) if the phone stream fails

## 7. Our own AI: the zone classifier
MAPPAL's zone recognition is a **MobileNetV2 we train ourselves** (transfer learning) on our own photos of the zones plus an `Other` class. The ArUco markers stay as the fallback. When the model is in `models/zone_model.pt`, the screen shows where the zone came from: `Shelf (AI 0.94)` or `Shelf (marker)`.

**Step 1: collect photos** (three separate sessions, different light / angle / time):
```bash
python tools/collect_dataset.py walk1      # training   (150+ photos per class)
python tools/collect_dataset.py walk2      # validation (100+ per class)
python tools/collect_dataset.py test       # final exam (100+ per class), never trained on
```
Keys: `0`-`3` = Desk, Shelf, Door, Window; `4` = Other; `SPACE` = record / pause; `q` = quit.
`Other` = walls, ceiling, floor, hallway, blurry motion, a hand over the lens. Move slowly, no faces.

**Step 2: train on Google Colab** (Runtime → Change runtime type → GPU). Upload `dataset.zip` and `training/train_zone.py`, then in a cell:
```
!unzip -q dataset.zip
!python train_zone.py --data dataset --train walk1 --val walk2 --test test --out models/zone_model.pt
!python train_zone.py --data dataset --train walk1 --val walk2 --test test --out models/zone_model_noreg.pt --no-reg
!zip -qr results.zip results models/zone_model.pt
```
Download `results.zip`, put `zone_model.pt` into `models/` and the `results/` folders into `results/`. (It also runs on a laptop CPU, just slower.)
The two runs are the regularisation evidence: without regularisation, train accuracy is near 100% but test accuracy drops. That gap is overfitting.

**Step 3: evaluate on this laptop:**
```bash
python training/evaluate.py --model models/zone_model.pt --test dataset/test
```
Writes `results/zone_model/` (confusion matrix, `failures.png`, `eval.json`) and `results/RESULTS.md`. Look at `failures.png` and write the "what went wrong" words in `RESULTS.md`.
Targets: test accuracy above 90%, under ~100 ms per image on CPU. 99%+ on test is suspicious: the test session probably looks too much like training.

**Honest limits:** YOLO is still pretrained; "Learns Normal" is a rule, not AI; the zone model only knows **this** room (~1000-2000 photos) and struggles with big light / angle changes.

## Troubleshooting
- **"Could not open camera source"**: check the phone URL in a browser first; both devices on the same Wi-Fi.
- **Marker not recognised**: more light, marker flat, fill more of the screen with it.
- **Objects missing from the list**: YOLO needs the object clearly visible; objects must be COCO classes. Lower `MIN_CONFIDENCE` in `config.py` if needed.
- **Too slow**: set `YOLO_IMAGE_SIZE = 320` in `config.py`.

## Project structure
```
config.py              all settings (camera, zones, thresholds)
main.py                the live app
tools/make_markers.py  printable ArUco markers
tools/zone_viewer.py   live marker test
tools/benchmark_detector.py  CPU speed test
zones/aruco_zone.py    marker id -> zone name
vision/camera.py       webcam / phone stream
vision/snapshot.py     3 sharpest frames per zone visit
vision/detector.py     YOLO nano -> inventory dict
vision/zone_classifier.py  our trained zone model + ZoneSmoother
tools/collect_dataset.py   collect our own zone photos
training/train_zone.py     train the zone classifier (Colab or CPU)
training/evaluate.py       test-session evaluation + results/RESULTS.md
memory/db.py           SQLite: zones, visits, inventories, events
memory/change_engine.py   what changed since the last visit
memory/normal_learner.py  routine vs alert
map/topo_map.py        live topological map (OpenCV)
demo/pixel_baseline.py "dumb" pixel-diff system for Act 3
tests/                 pytest, no camera or model needed
```
