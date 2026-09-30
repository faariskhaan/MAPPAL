# MAPPAL - the walking memory 
Final Year Project. Software prototype only. No hardware.

*Cameras see pixels. MAPPAL's AI sees meaning.*

## Install (Windows, CPU only)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Week 1 demo
1. Print the zone markers: `python tools/make_markers.py`, then print the PNGs in `markers/`.
2. Check zone recognition: `python tools/zone_viewer.py`
3. Speed test: `python tools/benchmark_detector.py`
4. Run MAPPAL: `python main.py`
   - Phone camera (IP Webcam / DroidCam, same Wi-Fi): `python main.py --source http://192.168.x.x:8080/video`
   - Point the camera at a marker for ~2 s: the screen shows `Zone: Shelf` and the object list, and the visit is saved in `mappal.db`.
   - Press `q` or `Esc` to quit.

## Week 2 demo (memory + map)
1. Run `python main.py` (or with `--source <phone url>`).
2. **Walk 1 (learning):** visit each zone marker for ~2 s. The live map on the right builds as you walk: every zone is a box, every walk between two zones is a line.
3. Remove an object (e.g. the backpack on the Shelf).
4. **Walk 2:** wait 5 s away from the zone, then visit it again. The screen shows `ALERT  SHELF: backpack missing` and the Shelf box turns orange.
   - Blue = no change, orange = changed, grey = not visited for 2 minutes.
5. To start with an empty memory, delete `mappal.db`.

## Tests
```bash
python -m pytest
```
