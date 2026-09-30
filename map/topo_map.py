"""Live topological map: zones are nodes, walking from one zone to the next is an edge.

This is real mapping without SLAM: it answers "which places exist and how are they
connected", not "where is every wall in centimetres". Humans remember buildings
the same way ("the shelf is next to the door").

Node colours:
  blue   = last visit found no change
  orange = last visit found an unusual change (alert)
  peach  = last visit only found routine changes (things that usually change here)
  grey   = not visited for STALE_SECONDS (the memory may be out of date)
"""

import math
import time

import cv2
import numpy as np

import config

# Colours are BGR (OpenCV order).
BLUE = (200, 120, 30)
ORANGE = (0, 140, 255)
PEACH = (150, 200, 250)   # softer colour for routine changes
GREY = (130, 130, 130)
WHITE = (255, 255, 255)
EDGE = (200, 200, 200)
BACKGROUND = (30, 30, 30)

NODE_W, NODE_H = 130, 50
TITLE_H, ALERT_H = 45, 70


class TopoMap:
    def __init__(self, clock=time.monotonic):
        self.clock = clock      # injectable so tests can fake time
        self.nodes = []         # zone names in order of discovery
        self.edges = set()      # {("Desk", "Shelf"), ...} - names sorted inside each pair
        self.last_seen = {}     # zone -> time it was last visited
        self.changed = {}       # zone -> None, "routine" or "alert" (result of its last visit)
        self.current = None     # the zone the camera is in right now
        self.last_alert = ""    # text shown under the map

    def visit(self, zone):
        """The camera is in `zone` now. Adds the node, and an edge if we walked here from another zone."""
        if zone not in self.nodes:
            self.nodes.append(zone)
            self.changed[zone] = None
        if self.current is not None and self.current != zone:
            self.edges.add(tuple(sorted((self.current, zone))))
        self.current = zone
        self.last_seen[zone] = self.clock()

    def set_result(self, zone, changes):
        """Record the outcome of a zone's analysis (list of changes from the change engine)."""
        if zone not in self.nodes:
            self.visit(zone)
        if any(c.get("severity", "alert") == "alert" for c in changes):
            self.changed[zone] = "alert"
        elif changes:
            self.changed[zone] = "routine"
        else:
            self.changed[zone] = None

    def set_alert(self, text):
        self.last_alert = text

    def node_color(self, zone, now=None):
        now = self.clock() if now is None else now
        if now - self.last_seen.get(zone, now) >= config.STALE_SECONDS:
            return GREY
        return {"alert": ORANGE, "routine": PEACH}.get(self.changed.get(zone), BLUE)

    def layout(self, width, height):
        """Node centre positions: evenly around a circle, clockwise from the top, in discovery order."""
        area_top, area_bottom = TITLE_H, height - ALERT_H
        cx, cy = width // 2, (area_top + area_bottom) // 2
        n = len(self.nodes)
        if n == 1:
            return {self.nodes[0]: (cx, cy)}
        radius = max(min(width - NODE_W, area_bottom - area_top - NODE_H) // 2 - 10, 20)
        positions = {}
        for i, zone in enumerate(self.nodes):
            angle = -math.pi / 2 + 2 * math.pi * i / n
            positions[zone] = (int(cx + radius * math.cos(angle)), int(cy + radius * math.sin(angle)))
        return positions

    def draw(self, width, height):
        """Return the map as an OpenCV image (height x width x 3)."""
        img = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
        _text(img, "MAPPAL live map", (15, 30), 0.8, WHITE, 2)
        _draw_legend(img, width)

        if not self.nodes:
            _text(img, "Walk to a zone marker...", (15, height // 2), 0.7, GREY, 1)
        pos = self.layout(width, height)
        for a, b in self.edges:
            cv2.line(img, pos[a], pos[b], EDGE, 2, cv2.LINE_AA)
        now = self.clock()
        for zone in self.nodes:
            x, y = pos[zone]
            x1, y1, x2, y2 = x - NODE_W // 2, y - NODE_H // 2, x + NODE_W // 2, y + NODE_H // 2
            _rounded_box(img, (x1, y1), (x2, y2), 12, self.node_color(zone, now), -1)
            if zone == self.current:
                _rounded_box(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 14, WHITE, 2)
            _centered_text(img, zone, (x, y), NODE_W - 16)

        # Last alert under the map.
        cv2.line(img, (0, height - ALERT_H), (width, height - ALERT_H), (70, 70, 70), 1)
        _text(img, "Last alert:", (15, height - ALERT_H + 25), 0.55, GREY, 1)
        _text(img, self.last_alert or "none", (15, height - 15), 0.7,
              ORANGE if self.last_alert else GREY, 2)
        return img


# --- small drawing helpers ------------------------------------------------

def _text(img, text, org, scale, color, thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _centered_text(img, text, center, max_width):
    """White zone name centred in its box, shrunk if it is too long."""
    scale = 0.7
    while scale > 0.35:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if w <= max_width:
            break
        scale -= 0.05
    _text(img, text, (center[0] - w // 2, center[1] + h // 2), scale, WHITE, 2)


def _rounded_box(img, top_left, bottom_right, r, color, thickness):
    """Rectangle with rounded corners. thickness=-1 fills it."""
    (x1, y1), (x2, y2) = top_left, bottom_right
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_legend(img, width):
    x = width - 305
    for label, color in (("ok", BLUE), ("routine", PEACH), ("alert", ORANGE), ("stale", GREY)):
        cv2.circle(img, (x, 24), 7, color, -1, cv2.LINE_AA)
        _text(img, label, (x + 12, 30), 0.5, WHITE, 1)
        x += 30 + len(label) * 10
