"""Tests for main.py logic: cooldown rule and one visit end-to-end (no camera, no model)."""

import numpy as np
import pytest

from main import alert_text, compose, process_visit, should_process
from memory import db


@pytest.fixture
def fresh_db(tmp_path):
    db.init_db(str(tmp_path / "test.db"))
    yield
    db.close_db()


def test_new_zone_is_processed():
    assert should_process("Shelf", now=10.0, last_seen={}, cooldown=5)


def test_zone_still_in_view_is_not_reprocessed():
    assert not should_process("Shelf", now=10.0, last_seen={"Shelf": 9.9}, cooldown=5)


def test_zone_back_after_cooldown_is_processed():
    assert should_process("Shelf", now=16.0, last_seen={"Shelf": 10.0}, cooldown=5)


def test_other_zones_do_not_affect_cooldown():
    assert should_process("Desk", now=10.0, last_seen={"Shelf": 9.9}, cooldown=5)


def test_two_walks_detect_missing_backpack(fresh_db):
    """The Week 2 demo: first walk learns the shelf, second walk notices the backpack is gone."""
    _, first = process_visit("Shelf", {"backpack": 1, "bottle": 2})
    assert first == []                                   # first visit: nothing to compare
    _, second = process_visit("Shelf", {"bottle": 2})
    assert [(c["object"], c["type"]) for c in second] == [("backpack", "missing")]
    assert db.get_last_inventory("Shelf") == {"bottle": 2}
    assert db.get_events("Shelf") == second              # the change is stored


def test_unchanged_visit_saves_no_events(fresh_db):
    process_visit("Desk", {"laptop": 1})
    _, changes = process_visit("Desk", {"laptop": 1})
    assert changes == []
    assert db.get_events() == []
    assert len(db.get_visit_history("Desk")) == 2


def test_alert_text():
    changes = [
        {"zone": "Shelf", "object": o, "type": "missing", "before": 1, "after": 0,
         "severity": "alert"}
        for o in ("backpack", "bottle", "cup")
    ]
    assert alert_text([]) == ""
    assert alert_text(changes[:1]) == "SHELF: backpack missing"
    assert alert_text(changes) == "SHELF: backpack missing | SHELF: bottle missing  (+1 more)"


def test_compose_puts_camera_left_and_map_right():
    cam = np.zeros((540, 720, 3), np.uint8)
    topo_img = np.full((540, 540, 3), 255, np.uint8)
    out = compose(cam, topo_img)
    assert out.shape == (540, 1260, 3)
    assert out[:, :720].max() == 0 and out[:, 720:].min() == 255
