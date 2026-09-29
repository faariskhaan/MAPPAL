"""Detector logic tests with fake detections (the YOLO model is never loaded)."""

from vision.detector import combine_snapshots, count_objects


def test_count_objects_applies_confidence_and_ignores_person():
    names = ["bottle", "bottle", "person", "cup", "bottle"]
    confs = [0.9, 0.6, 0.95, 0.3, 0.5]
    assert count_objects(names, confs, 0.5, {"person"}) == {"bottle": 3}


def test_object_must_appear_in_two_snapshots():
    shots = [{"bottle": 1, "cup": 1}, {"bottle": 1}, {"bottle": 1, "book": 1}]
    assert combine_snapshots(shots, min_votes=2) == {"bottle": 1}


def test_count_is_median_of_snapshots_where_seen():
    shots = [{"bottle": 3}, {"bottle": 2}, {"bottle": 2}]
    assert combine_snapshots(shots) == {"bottle": 2}
    shots = [{"book": 1}, {"book": 4}, {"book": 2}]
    assert combine_snapshots(shots) == {"book": 2}


def test_two_votes_even_split_uses_lower_median():
    shots = [{"chair": 1}, {"chair": 2}, {}]
    assert combine_snapshots(shots) == {"chair": 1}


def test_nothing_seen_gives_empty_inventory():
    assert combine_snapshots([{}, {}, {}]) == {}
    assert combine_snapshots([]) == {}


def test_single_snapshot_still_works():
    assert combine_snapshots([{"laptop": 1}], min_votes=2) == {"laptop": 1}
