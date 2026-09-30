"""SQLite memory for MAPPAL.

Four tables:
  zones        - one row per zone name ("Desk", "Shelf", ...)
  visits       - one row every time the camera visits a zone
  inventories  - the objects seen on a visit (one row per object class)
  events       - the changes found between two visits (used from Week 2)

Timestamps are Unix seconds (time.time()), so "how long ago" is a simple subtraction.
Call init_db(path) once at start-up; every other function uses that connection.
"""

import sqlite3
import time

_conn = None  # the open database connection, set by init_db()

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS visits (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id   INTEGER NOT NULL REFERENCES zones(id),
    timestamp REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS inventories (
    visit_id INTEGER NOT NULL REFERENCES visits(id),
    object   TEXT NOT NULL,
    count    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id   INTEGER NOT NULL REFERENCES zones(id),
    timestamp REAL NOT NULL,
    object    TEXT NOT NULL,
    type      TEXT NOT NULL,
    "before"  INTEGER NOT NULL,
    "after"   INTEGER NOT NULL,
    severity  TEXT NOT NULL
);
"""


def init_db(path):
    """Open (or create) the database file and make sure all tables exist."""
    global _conn
    if _conn is not None:
        _conn.close()
    _conn = sqlite3.connect(path)
    _conn.executescript(SCHEMA)
    _conn.commit()
    return _conn


def reset_db():
    """Forget everything (the demo's "Reset memory" key). The tables stay, the rows go."""
    for table in ("inventories", "events", "visits", "zones"):
        _db().execute(f"DELETE FROM {table}")
    _db().commit()


def close_db():
    """Close the connection (used by tests and at app exit)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def _db():
    if _conn is None:
        raise RuntimeError("Database not opened. Call init_db(path) first.")
    return _conn


def _zone_id(zone_name, create=True):
    """Return the id of a zone. Creates the zone row the first time it is seen."""
    row = _db().execute("SELECT id FROM zones WHERE name = ?", (zone_name,)).fetchone()
    if row:
        return row[0]
    if not create:
        return None
    cur = _db().execute("INSERT INTO zones (name) VALUES (?)", (zone_name,))
    return cur.lastrowid


def _inventory_of(visit_id):
    rows = _db().execute(
        "SELECT object, count FROM inventories WHERE visit_id = ? ORDER BY object",
        (visit_id,),
    ).fetchall()
    return {obj: count for obj, count in rows}


def save_visit(zone_name, inventory, timestamp=None):
    """Store one visit and its inventory. Returns the new visit id."""
    if timestamp is None:
        timestamp = time.time()
    zone_id = _zone_id(zone_name)
    cur = _db().execute(
        "INSERT INTO visits (zone_id, timestamp) VALUES (?, ?)", (zone_id, timestamp)
    )
    visit_id = cur.lastrowid
    _db().executemany(
        "INSERT INTO inventories (visit_id, object, count) VALUES (?, ?, ?)",
        [(visit_id, obj, int(count)) for obj, count in inventory.items()],
    )
    _db().commit()
    return visit_id


def get_last_inventory(zone_name):
    """Inventory of the most recent visit, or None if the zone was never visited.

    Note: a visit where nothing was seen returns {} (empty dict), not None.
    """
    zone_id = _zone_id(zone_name, create=False)
    if zone_id is None:
        return None
    row = _db().execute(
        "SELECT id FROM visits WHERE zone_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
        (zone_id,),
    ).fetchone()
    if row is None:
        return None
    return _inventory_of(row[0])


def get_visit_history(zone_name):
    """All inventories of a zone, oldest visit first."""
    zone_id = _zone_id(zone_name, create=False)
    if zone_id is None:
        return []
    rows = _db().execute(
        "SELECT id FROM visits WHERE zone_id = ? ORDER BY timestamp, id", (zone_id,)
    ).fetchall()
    return [_inventory_of(visit_id) for (visit_id,) in rows]


def save_events(changes, timestamp=None):
    """Store a list of change dicts (format in CLAUDE.md)."""
    if timestamp is None:
        timestamp = time.time()
    for c in changes:
        _db().execute(
            'INSERT INTO events (zone_id, timestamp, object, type, "before", "after", severity) '
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _zone_id(c["zone"]),
                timestamp,
                c["object"],
                c["type"],
                c["before"],
                c["after"],
                c["severity"],
            ),
        )
    _db().commit()


def get_events(zone_name=None):
    """Stored events as change dicts, oldest first. Optionally only for one zone."""
    sql = (
        'SELECT z.name, e.object, e.type, e."before", e."after", e.severity '
        "FROM events e JOIN zones z ON z.id = e.zone_id"
    )
    params = ()
    if zone_name is not None:
        sql += " WHERE z.name = ?"
        params = (zone_name,)
    sql += " ORDER BY e.timestamp, e.id"
    keys = ("zone", "object", "type", "before", "after", "severity")
    return [dict(zip(keys, row)) for row in _db().execute(sql, params).fetchall()]


def get_last_visit_time(zone_name):
    """Unix time of the latest visit to a zone, or None if never visited."""
    zone_id = _zone_id(zone_name, create=False)
    if zone_id is None:
        return None
    row = _db().execute(
        "SELECT MAX(timestamp) FROM visits WHERE zone_id = ?", (zone_id,)
    ).fetchone()
    return row[0]
