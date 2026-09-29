"""Test the zone cooldown rule of main.py (no camera, no model)."""

from main import should_process


def test_new_zone_is_processed():
    assert should_process("Shelf", now=10.0, last_seen={}, cooldown=5)


def test_zone_still_in_view_is_not_reprocessed():
    assert not should_process("Shelf", now=10.0, last_seen={"Shelf": 9.9}, cooldown=5)


def test_zone_back_after_cooldown_is_processed():
    assert should_process("Shelf", now=16.0, last_seen={"Shelf": 10.0}, cooldown=5)


def test_other_zones_do_not_affect_cooldown():
    assert should_process("Desk", now=10.0, last_seen={"Shelf": 9.9}, cooldown=5)
