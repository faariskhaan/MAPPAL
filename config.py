"""MAPPAL settings. Every number the team may want to tweak lives here."""

# --- Camera -------------------------------------------------------------
# 0 = laptop webcam. For a phone use the stream URL shown by the app, e.g.
#   IP Webcam: "http://192.168.1.23:8080/video"
#   DroidCam:  "http://192.168.1.23:4747/video"
# (You can also override it at run time: python main.py --source <url>)
CAMERA_SOURCE = 0

# --- Zones (ArUco) ------------------------------------------------------
ARUCO_DICT = "DICT_4X4_50"   # name of the OpenCV ArUco dictionary
ZONES = {0: "Desk", 1: "Shelf", 2: "Door", 3: "Window"}

# --- Snapshots ----------------------------------------------------------
SNAPSHOT_SECONDS = 2         # how long to look at a zone when it is first seen
SNAPSHOT_COUNT = 3           # how many of the sharpest frames go to the AI

# --- AI detector (YOLO nano, CPU) ---------------------------------------
YOLO_MODEL = "yolov8n.pt"    # downloaded automatically the first time (~6 MB)
YOLO_IMAGE_SIZE = 640        # use 320 if the laptop is too slow
MIN_CONFIDENCE = 0.5         # ignore detections below this confidence
IGNORED_CLASSES = {"person"} # people walk around; they are not part of a zone
MIN_SNAPSHOT_VOTES = 2       # an object must be seen in at least this many snapshots

# --- Main app -----------------------------------------------------------
ZONE_COOLDOWN = 5            # seconds a zone must be out of view before it is scanned again
DB_PATH = "mappal.db"        # SQLite memory file (not committed to git)

# --- Live map -----------------------------------------------------------
STALE_SECONDS = 120          # a zone turns grey if not visited for this long

# --- Window -------------------------------------------------------------
VIEW_HEIGHT = 540            # height of the app window (camera is resized to this)
MAP_WIDTH = 540              # width of the map panel on the right
ALERT_SECONDS = 4            # how long the orange alert banner stays on screen
DEMO_UI_SCALE = 1.4          # --demo: text and boxes this many times bigger

# --- Learns normal ------------------------------------------------------
MIN_VISITS = 3               # visits of history needed before anything can be "routine"
ROUTINE_RATIO = 0.5          # changed in more than this share of visits = routine

# --- Pixel baseline (demo Act 3, python main.py --compare) --------------
PIXEL_THRESHOLD = 30         # a pixel "changed" if its grey value moved more than this
PIXEL_CHANGED_PERCENT = 10   # the pixel system says CHANGED above this % of pixels

# --- Zone classifier dataset (tools/collect_dataset.py) -----------------
EXTRA_CLASS = "Other"        # "not in any zone": walls, floor, hallway, blur, hand on lens
SAVE_INTERVAL = 0.3          # seconds between saved frames while recording
DATASET_DIR = "dataset"      # photos go to dataset/<session>/<class>/ (not committed)

# --- Zone classifier in the app (our trained MobileNetV2) ---------------
ZONE_MODEL_PATH = "models/zone_model.pt"
ZONE_CONF = 0.85             # the AI must be at least this sure...
ZONE_CONSECUTIVE = 5         # ...this many predictions in a row before a zone counts
CLASSIFY_EVERY = 5           # run the classifier on every 5th frame only (CPU budget)
