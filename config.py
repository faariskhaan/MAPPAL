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

