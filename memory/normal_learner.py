"""Normal learner: MAPPAL learns what usually changes in a zone ("right thing at the right time").

Example: the cup on the Desk moves on almost every visit -> that is normal, a "routine"
change. The backpack on the Shelf has been there on every visit -> if it goes missing,
that is unusual, an "alert".

How: look at the zone's past visits in order and count, for each object, in how many
visit-to-visit steps its count changed. If it changed in more than ROUTINE_RATIO of the
steps, and we have at least MIN_VISITS visits of history, its changes are "routine".
Otherwise (or when MAPPAL has not seen enough visits yet) they are "alert".
"""

import config


def change_rates(history):
    """For each object: the fraction of visit-to-visit steps in which its count changed.

    history is a list of inventories, oldest first. 3 visits = 2 steps.
    """
    steps = list(zip(history, history[1:]))
    if not steps:
        return {}
    objects = set().union(*history)
    return {
        obj: sum(1 for before, after in steps if before.get(obj, 0) != after.get(obj, 0)) / len(steps)
        for obj in objects
    }


def severity(obj, history, min_visits=None, routine_ratio=None):
    """'routine' if this object changes often in this zone, otherwise 'alert'."""
    min_visits = config.MIN_VISITS if min_visits is None else min_visits
    routine_ratio = config.ROUTINE_RATIO if routine_ratio is None else routine_ratio
    if len(history) < min_visits:
        return "alert"            # not enough experience yet: better safe than sorry
    return "routine" if change_rates(history).get(obj, 0.0) > routine_ratio else "alert"


def assign_severity(changes, history, min_visits=None, routine_ratio=None):
    """Return copies of the changes with their 'severity' filled in from the history."""
    return [
        {**c, "severity": severity(c["object"], history, min_visits, routine_ratio)}
        for c in changes
    ]
