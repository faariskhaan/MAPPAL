"""Training helper tests: class discovery, session split, metrics (no training, no downloads)."""

import numpy as np
import pytest

from training.train_zone import (
    SessionDataset, build_model, confusion, discover_classes, eval_transform, per_class_metrics,
    set_phase,
)


def _make_session(root, session, classes, n=2):
    import cv2
    for c in classes:
        d = root / session / c
        d.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(d / f"{c}_{i}.jpg"), np.full((60, 80, 3), 100, np.uint8))


def test_classes_follow_zone_order_then_extras(tmp_path):
    _make_session(tmp_path, "walk1", ["Other", "Window", "Desk", "Shelf", "Door"])
    _make_session(tmp_path, "walk2", ["Desk", "Hallway"])
    assert discover_classes(tmp_path, ["walk1", "walk2"]) == [
        "Desk", "Shelf", "Door", "Window", "Other", "Hallway"]


def test_sessions_stay_separate(tmp_path):
    _make_session(tmp_path, "walk1", ["Desk", "Shelf"], n=3)
    _make_session(tmp_path, "test", ["Desk", "Shelf"], n=2)
    names = ["Desk", "Shelf"]
    train = SessionDataset(tmp_path, ["walk1"], names, eval_transform())
    test = SessionDataset(tmp_path, ["test"], names, eval_transform())
    assert len(train) == 6 and len(test) == 2 * 2
    assert not {p for p, _ in train.samples} & {p for p, _ in test.samples}
    x, y = test[0]
    assert tuple(x.shape) == (3, 224, 224) and y == 0


def test_missing_session_stops_with_a_clear_message(tmp_path):
    with pytest.raises(SystemExit):
        discover_classes(tmp_path, ["nope"])


def test_confusion_and_per_class_metrics():
    cm = confusion([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], 2)
    assert cm.tolist() == [[1, 1], [1, 2]]
    m = per_class_metrics(cm, ["Desk", "Shelf"])
    assert m["Desk"]["precision"] == 0.5 and m["Desk"]["recall"] == 0.5
    assert m["Shelf"]["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert m["Shelf"]["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert m["Shelf"]["support"] == 3


def test_phase_1_trains_only_the_head_phase_2_also_last_blocks():
    model = build_model(5, dropout=0.3, pretrained=False)
    set_phase(model, 1)
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert not any(p.requires_grad for p in model.features.parameters())
    set_phase(model, 2)
    assert all(p.requires_grad for p in model.features[-2:].parameters())
    assert not any(p.requires_grad for p in model.features[:-2].parameters())
