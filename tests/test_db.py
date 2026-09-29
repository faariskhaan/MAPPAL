"""Tests for memory/db.py. They use a temporary database file, no camera, no YOLO."""

import pytest

from memory import db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    db.init_db(str(tmp_path / "test.db"))
    yield
    db.close_db()


def test_unknown_zone_has_no_memory():
    assert db.get_last_inventory("Shelf") is None
    assert db.get_visit_history("Shelf") == []
    assert db.get_last_visit_time("Shelf") is None


def test_save_visit_returns_ids_and_stores_inventory():
    v1 = db.save_visit("Shelf", {"bottle": 2, "backpack": 1}, timestamp=100.0)
    v2 = db.save_visit("Desk", {"laptop": 1}, timestamp=101.0)
    assert v1 != v2
    assert db.get_last_inventory("Shelf") == {"bottle": 2, "backpack": 1}
    assert db.get_last_inventory("Desk") == {"laptop": 1}


def test_last_inventory_is_the_newest_visit():
    db.save_visit("Shelf", {"bottle": 2}, timestamp=100.0)
    db.save_visit("Shelf", {"bottle": 1, "cup": 1}, timestamp=200.0)
    assert db.get_last_inventory("Shelf") == {"bottle": 1, "cup": 1}


def test_empty_visit_is_empty_dict_not_none():
    db.save_visit("Door", {}, timestamp=100.0)
    assert db.get_last_inventory("Door") == {}


def test_visit_history_oldest_first():
    db.save_visit("Shelf", {"bottle": 2}, timestamp=100.0)
    db.save_visit("Desk", {"laptop": 1}, timestamp=150.0)
    db.save_visit("Shelf", {"bottle": 1}, timestamp=200.0)
    db.save_visit("Shelf", {}, timestamp=300.0)
    assert db.get_visit_history("Shelf") == [{"bottle": 2}, {"bottle": 1}, {}]


def test_last_visit_time():
    db.save_visit("Window", {"chair": 1}, timestamp=100.0)
    db.save_visit("Window", {"chair": 1}, timestamp=250.5)
    assert db.get_last_visit_time("Window") == 250.5


def test_save_and_read_events():
    changes = [
        {"zone": "Shelf", "object": "backpack", "type": "missing",
         "before": 1, "after": 0, "severity": "alert"},
        {"zone": "Desk", "object": "cup", "type": "new",
         "before": 0, "after": 1, "severity": "routine"},
    ]
    db.save_events(changes, timestamp=500.0)
    assert db.get_events() == changes
    assert db.get_events("Shelf") == [changes[0]]


def test_data_survives_reopening(tmp_path):
    path = str(tmp_path / "persist.db")
    db.init_db(path)
    db.save_visit("Shelf", {"book": 3}, timestamp=1.0)
    db.close_db()
    db.init_db(path)
    assert db.get_last_inventory("Shelf") == {"book": 3}


def test_using_db_before_init_raises():
    db.close_db()
    with pytest.raises(RuntimeError):
        db.save_visit("Shelf", {})
