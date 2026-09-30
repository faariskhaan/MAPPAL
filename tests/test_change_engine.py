"""Change engine tests with fake inventories (no camera, no YOLO)."""

from memory.change_engine import compare, describe


def _types(changes):
    return {c["object"]: c["type"] for c in changes}


def test_first_visit_has_no_changes():
    assert compare("Shelf", None, {"bottle": 2}) == []
    assert compare("Shelf", None, {}) == []


def test_identical_inventories_have_no_changes():
    inv = {"bottle": 2, "backpack": 1}
    assert compare("Shelf", inv, dict(inv)) == []


def test_empty_to_empty_has_no_changes():
    assert compare("Door", {}, {}) == []


def test_missing_object():
    changes = compare("Shelf", {"bottle": 2, "backpack": 1}, {"bottle": 2})
    assert changes == [{
        "zone": "Shelf", "object": "backpack", "type": "missing",
        "before": 1, "after": 0, "severity": "alert",
    }]


def test_new_object():
    changes = compare("Desk", {"laptop": 1}, {"laptop": 1, "cup": 1})
    assert changes == [{
        "zone": "Desk", "object": "cup", "type": "new",
        "before": 0, "after": 1, "severity": "alert",
    }]


def test_count_changed_up_and_down():
    changes = compare("Shelf", {"bottle": 3, "book": 1}, {"bottle": 1, "book": 4})
    assert _types(changes) == {"bottle": "count_changed", "book": "count_changed"}
    by_obj = {c["object"]: c for c in changes}
    assert (by_obj["bottle"]["before"], by_obj["bottle"]["after"]) == (3, 1)
    assert (by_obj["book"]["before"], by_obj["book"]["after"]) == (1, 4)


def test_everything_gone():
    changes = compare("Window", {"chair": 2, "cup": 1}, {})
    assert _types(changes) == {"chair": "missing", "cup": "missing"}
    assert all(c["after"] == 0 for c in changes)


def test_everything_new_after_empty_visit():
    changes = compare("Door", {}, {"backpack": 1})
    assert _types(changes) == {"backpack": "new"}


def test_mixed_changes_sorted_by_object_name():
    old = {"bottle": 2, "backpack": 1, "cup": 1}
    new = {"bottle": 1, "cup": 1, "laptop": 1}
    changes = compare("Shelf", old, new)
    assert [c["object"] for c in changes] == ["backpack", "bottle", "laptop"]
    assert _types(changes) == {
        "backpack": "missing", "bottle": "count_changed", "laptop": "new",
    }


def test_zero_counts_are_treated_as_absent():
    assert compare("Shelf", {"bottle": 0}, {}) == []
    assert _types(compare("Shelf", {"cup": 0}, {"cup": 1})) == {"cup": "new"}


def test_every_change_is_an_alert_with_the_zone_name():
    changes = compare("Shelf", {"bottle": 1}, {"cup": 1})
    assert all(c["severity"] == "alert" and c["zone"] == "Shelf" for c in changes)


def test_inputs_are_not_modified():
    old, new = {"bottle": 2}, {"cup": 1}
    compare("Shelf", old, new)
    assert old == {"bottle": 2} and new == {"cup": 1}


def test_describe():
    assert describe({"zone": "Shelf", "object": "backpack", "type": "missing",
                     "before": 1, "after": 0, "severity": "alert"}) == "SHELF: backpack missing"
    assert describe({"zone": "Desk", "object": "cup", "type": "new",
                     "before": 0, "after": 1, "severity": "alert"}) == "DESK: new cup"
    assert describe({"zone": "Shelf", "object": "bottle", "type": "count_changed",
                     "before": 2, "after": 1, "severity": "alert"}) == "SHELF: bottle 2 -> 1"
