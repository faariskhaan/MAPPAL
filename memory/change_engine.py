"""Change engine: compare a zone's last inventory with the new one and list what changed.

A change is a dict (format fixed in CLAUDE.md):
  {"zone": "Shelf", "object": "backpack", "type": "missing", "before": 1, "after": 0,
   "severity": "alert"}

Types:
  missing        - was there last time, now gone        (before > 0, after = 0)
  new            - was not there last time, now present (before = 0, after > 0)
  count_changed  - still there, but a different number  (before > 0, after > 0)

Severity:
  alert    - unusual change, shown in orange with a banner
  routine  - this object changes often here (learned by normal_learner), shown softly
"""

from memory.normal_learner import assign_severity


def compare(zone, old_inventory, new_inventory, history=None):
    """Return the list of changes between two inventories (sorted by object name).

    old_inventory is None on the first visit of a zone: nothing to compare yet,
    so we return an empty list (the first visit only teaches MAPPAL what is there).
    history = the zone's past inventories (oldest first). When given, the normal
    learner decides each change's severity; without it every change is an "alert".
    """
    if old_inventory is None:
        return []

    changes = []
    for obj in sorted(set(old_inventory) | set(new_inventory)):
        before = old_inventory.get(obj, 0)
        after = new_inventory.get(obj, 0)
        if before == after:
            continue
        if after == 0:
            change_type = "missing"
        elif before == 0:
            change_type = "new"
        else:
            change_type = "count_changed"
        changes.append({
            "zone": zone,
            "object": obj,
            "type": change_type,
            "before": before,
            "after": after,
            "severity": "alert",
        })
    if history is not None:
        changes = assign_severity(changes, history)
    return changes


def describe(change):
    """One short line for the screen, e.g. 'SHELF: backpack missing'."""
    zone = change["zone"].upper()
    obj = change["object"]
    if change["type"] == "missing":
        return f"{zone}: {obj} missing"
    if change["type"] == "new":
        return f"{zone}: new {obj}"
    return f"{zone}: {obj} {change['before']} -> {change['after']}"
