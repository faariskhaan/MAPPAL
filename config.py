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

