"""Evaluation helpers: failure grid, confusions and the RESULTS.md report (no model needed)."""

import json

import cv2
import numpy as np

from training.evaluate import failure_grid, top_confusions, write_report


def test_failure_grid_layout(tmp_path):
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), np.full((100, 100, 3), 50, np.uint8))
    grid = failure_grid([(str(p), 0, 1, 0.97)] * 5, ["Desk", "Shelf"])
    assert grid.shape == (2 * (180 + 44), 4 * 240, 3)          # 5 tiles -> 2 rows of 4
    assert failure_grid([], ["Desk"]).shape[0] > 0               # no failures still gives an image


def test_top_confusions():
    cm = np.array([[9, 1, 0], [4, 5, 0], [0, 2, 7]])
    assert top_confusions(cm, ["Desk", "Door", "Window"], n=2) == [
        ("Door", "Desk", 4), ("Window", "Door", 2)]


def test_report_has_both_runs_and_the_confusions(tmp_path):
    reg, noreg = tmp_path / "zone_model", tmp_path / "zone_model_noreg"
    reg.mkdir()
    noreg.mkdir()
    (reg / "metrics.json").write_text(json.dumps(
        {"regularised": True, "train_acc": 0.95, "val_acc": 0.93, "test_acc": 0.91}))
    (reg / "eval.json").write_text(json.dumps(
        {"accuracy": 0.91, "inference_ms": 23.4, "top_confusions": [["Door", "Window", 6]]}))
    (noreg / "metrics.json").write_text(json.dumps(
        {"regularised": False, "train_acc": 1.0, "val_acc": 0.84, "test_acc": 0.79}))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "| zone_model | with regularisation | 95.0% | 93.0% | 91.0% | 23.4 |" in text
    assert "| zone_model_noreg | without regularisation | 100.0% | 84.0% | 79.0% | - |" in text
    assert "**Door** were predicted as **Window**" in text
