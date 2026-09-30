"""Normal learner tests with fake visit histories (no camera, no YOLO)."""

from memory.change_engine import compare
from memory.normal_learner import assign_severity, change_rates, severity

# The Desk: the cup comes and goes on every visit, the laptop never moves.
DESK_HISTORY = [
    {"laptop": 1, "cup": 1},
    {"laptop": 1},
    {"laptop": 1, "cup": 1},
    {"laptop": 1},
]


def test_change_rates():
    rates = change_rates(DESK_HISTORY)
    assert rates == {"cup": 1.0, "laptop": 0.0}


def test_change_rates_need_two_visits():
    assert change_rates([]) == {}
    assert change_rates([{"cup": 1}]) == {}


def test_count_change_counts_as_a_change():
    history = [{"bottle": 1}, {"bottle": 2}, {"bottle": 2}]
    assert change_rates(history) == {"bottle": 0.5}


def test_often_changing_object_is_routine():
    assert severity("cup", DESK_HISTORY, min_visits=3, routine_ratio=0.5) == "routine"


def test_stable_object_is_alert():
    assert severity("laptop", DESK_HISTORY, min_visits=3, routine_ratio=0.5) == "alert"


def test_never_seen_object_is_alert():
    assert severity("backpack", DESK_HISTORY, min_visits=3, routine_ratio=0.5) == "alert"


def test_not_enough_visits_is_always_alert():
    short = DESK_HISTORY[:2]          # cup changed in 1 of 1 steps, but only 2 visits
    assert severity("cup", short, min_visits=3, routine_ratio=0.5) == "alert"


def test_exactly_the_ratio_is_not_routine():
    history = [{"cup": 1}, {}, {}]    # changed in 1 of 2 steps = 0.5, not MORE than 0.5
    assert severity("cup", history, min_visits=3, routine_ratio=0.5) == "alert"


def test_assign_severity_does_not_modify_input():
    changes = [{"zone": "Desk", "object": "cup", "type": "missing",
                "before": 1, "after": 0, "severity": "alert"}]
    result = assign_severity(changes, DESK_HISTORY, min_visits=3, routine_ratio=0.5)
    assert result[0]["severity"] == "routine"
    assert changes[0]["severity"] == "alert"


def test_change_engine_uses_the_learner():
    """Cup missing again = routine; laptop missing for the first time = alert."""
    changes = compare("Desk", DESK_HISTORY[-1] | {"cup": 1}, {}, history=DESK_HISTORY)
    by_obj = {c["object"]: c["severity"] for c in changes}
    assert by_obj == {"cup": "routine", "laptop": "alert"}


def test_change_engine_without_history_is_all_alert():
    changes = compare("Desk", {"cup": 1}, {})
    assert changes[0]["severity"] == "alert"
