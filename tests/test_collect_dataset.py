"""Dataset collector helpers (no camera needed)."""

import cv2
import numpy as np

from tools.collect_dataset import class_for_key, class_names, count_images, resize_longest, save_frame


def test_class_order_is_zone_ids_then_other():
    assert class_names() == ["Desk", "Shelf", "Door", "Window", "Other"]


def test_keys_select_classes():
    assert class_for_key(ord("0")) == "Desk"
    assert class_for_key(ord("3")) == "Window"
    assert class_for_key(ord("4")) == "Other"
    assert class_for_key(ord("5")) is None
    assert class_for_key(ord(" ")) is None


def test_resize_longest_side():
    assert resize_longest(np.zeros((1080, 1920, 3), np.uint8)).shape == (360, 640, 3)
    assert resize_longest(np.zeros((1920, 1080, 3), np.uint8)).shape == (640, 360, 3)
    assert resize_longest(np.zeros((240, 320, 3), np.uint8)).shape == (240, 320, 3)  # no enlarging


def test_save_frame_writes_jpeg_into_class_folder(tmp_path):
    folder = tmp_path / "walk1" / "Shelf"
    path = save_frame(np.full((720, 1280, 3), 128, np.uint8), str(folder), "Shelf")
    assert path.endswith(".jpg") and "Shelf_" in path
    assert cv2.imread(path).shape == (360, 640, 3)
    assert count_images(str(folder)) == 1
    assert count_images(str(tmp_path / "missing")) == 0
