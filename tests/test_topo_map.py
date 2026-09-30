"""Topological map tests with a fake clock (no camera)."""

import config
from map.topo_map import BLUE, GREY, ORANGE, TopoMap


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _walk(topo, *zones):
    for z in zones:
        topo.visit(z)


def test_nodes_in_order_of_discovery():
    topo = TopoMap(clock=FakeClock())
    _walk(topo, "Shelf", "Desk", "Shelf", "Door")
    assert topo.nodes == ["Shelf", "Desk", "Door"]


def test_edges_follow_the_walk():
    topo = TopoMap(clock=FakeClock())
    _walk(topo, "Desk", "Shelf", "Door", "Desk")
    assert topo.edges == {("Desk", "Shelf"), ("Door", "Shelf"), ("Desk", "Door")}


def test_staying_in_a_zone_adds_no_edge():
    topo = TopoMap(clock=FakeClock())
    _walk(topo, "Desk", "Desk", "Desk")
    assert topo.edges == set()


def test_walking_back_and_forth_gives_one_edge():
    topo = TopoMap(clock=FakeClock())
    _walk(topo, "Desk", "Shelf", "Desk", "Shelf")
    assert topo.edges == {("Desk", "Shelf")}


def test_colours_blue_orange_grey():
    clock = FakeClock()
    topo = TopoMap(clock=clock)
    _walk(topo, "Desk", "Shelf")
    topo.set_result("Desk", [])
    topo.set_result("Shelf", [{"object": "backpack"}])
    assert topo.node_color("Desk") == BLUE
    assert topo.node_color("Shelf") == ORANGE

    clock.now = config.STALE_SECONDS + 1
    topo.visit("Desk")                        # Desk seen again, Shelf not
    assert topo.node_color("Desk") == BLUE
    assert topo.node_color("Shelf") == GREY


def test_zone_turns_blue_again_after_a_clean_visit():
    topo = TopoMap(clock=FakeClock())
    topo.set_result("Shelf", [{"object": "backpack"}])
    assert topo.node_color("Shelf") == ORANGE
    topo.set_result("Shelf", [])
    assert topo.node_color("Shelf") == BLUE


def test_layout_positions_are_distinct_and_inside_the_image():
    topo = TopoMap(clock=FakeClock())
    _walk(topo, "Desk", "Shelf", "Door", "Window")
    pos = topo.layout(540, 540)
    assert len(set(pos.values())) == 4
    assert all(0 <= x < 540 and 0 <= y < 540 for x, y in pos.values())


def test_draw_returns_image_of_requested_size():
    topo = TopoMap(clock=FakeClock())
    assert topo.draw(400, 300).shape == (300, 400, 3)   # empty map still draws
    _walk(topo, "Desk", "Shelf", "Door")
    topo.set_result("Shelf", [{"object": "backpack"}])
    topo.set_alert("SHELF: backpack missing")
    img = topo.draw(540, 540)
    assert img.shape == (540, 540, 3)
    assert img.any()


def test_changed_node_is_drawn_orange():
    topo = TopoMap(clock=FakeClock())
    _walk(topo, "Shelf", "Desk")
    topo.set_result("Shelf", [{"object": "backpack"}])
    img = topo.draw(540, 540)
    x, y = topo.layout(540, 540)["Shelf"]
    # a pixel inside the box but away from the text
    assert tuple(int(v) for v in img[y - 20, x - 55]) == ORANGE


def test_routine_only_changes_are_peach_and_alert_wins():
    from map.topo_map import PEACH
    topo = TopoMap(clock=FakeClock())
    topo.set_result("Desk", [{"object": "cup", "severity": "routine"}])
    assert topo.node_color("Desk") == PEACH
    topo.set_result("Desk", [{"object": "cup", "severity": "routine"},
                             {"object": "laptop", "severity": "alert"}])
    assert topo.node_color("Desk") == ORANGE
