"""ZoneSmoother tests - no camera, no model, no torch."""

import sys

from vision.zone_classifier import ZoneSmoother


def _smoother():
    return ZoneSmoother(needed=5, min_conf=0.85, ignore="Other")


def _feed(s, label, n, conf=0.95):
    return [s.update(label, conf) for _ in range(n)]


def test_1_not_confirmed_before_5_consecutive_frames():
    assert _feed(_smoother(), "Shelf", 4) == [None] * 4


def test_2_confirmed_on_the_5th():
    assert _feed(_smoother(), "Shelf", 5)[-1] == "Shelf"


def test_3_a_different_label_resets_the_counter():
    s = _smoother()
    _feed(s, "Shelf", 4)
    assert s.update("Desk", 0.95) is None
    assert _feed(s, "Shelf", 4) == [None] * 4       # Shelf starts again from 1
    assert s.update("Shelf", 0.95) == "Shelf"


def test_4_low_confidence_never_confirms():
    s = _smoother()
    assert _feed(s, "Shelf", 20, conf=0.84) == [None] * 20
    _feed(s, "Shelf", 4)
    assert s.update("Shelf", 0.5) is None           # one unsure frame breaks the streak
    assert _feed(s, "Shelf", 4) == [None] * 4


def test_5_other_never_confirms():
    assert _feed(_smoother(), "Other", 20, conf=0.99) == [None] * 20


def test_6_after_confirming_a_new_zone_needs_5_new_frames():
    s = _smoother()
    assert _feed(s, "Shelf", 5)[-1] == "Shelf"
    assert s.update("Shelf", 0.95) == "Shelf"       # stays confirmed while it agrees
    assert _feed(s, "Door", 4) == [None] * 4
    assert s.update("Door", 0.95) == "Door"


def test_smoother_does_not_need_torch():
    import vision.zone_classifier  # noqa: F401
    assert "vision.zone_classifier" in sys.modules
    # the module must be importable without loading torch for the smoother
    src = open(vision.zone_classifier.__file__, encoding="utf-8").read()
    assert "\nimport torch" not in src
