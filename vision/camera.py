"""Open the camera: laptop webcam (0) or a phone stream URL."""

import cv2


def parse_source(source):
    """'0' -> 0 (webcam index). A URL or file path stays a string."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source)
    return source


def open_camera(source):
    """Open a webcam index or a phone stream URL. Raises if it cannot be opened."""
    source = parse_source(source)
    cap = cv2.VideoCapture(source)
    # Keep only the newest frame so a phone stream does not lag behind.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source {source!r}. "
            "Webcam: try 0 or 1. Phone: check the URL and that both are on the same Wi-Fi."
        )
    return cap
