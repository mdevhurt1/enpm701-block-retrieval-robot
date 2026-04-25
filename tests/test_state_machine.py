import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import numpy as np

from fp import state_machine
from fp.fsm import states as fsm_states


class _PlannerWithCollapsedSnap:
    def __init__(self):
        self.logged_traversed = []

    def get_next_goal(self, target_color, robot_world):
        return (142.5, 107.5)

    def astar(self, start_world, goal_world):
        return []

    def get_last_astar_debug(self):
        return {
            "planned_goal_world": (142.5, 77.5),
            "first_waypoint_world": None,
        }

    def log_traversed(self, x, y, radius_cm=None):
        self.logged_traversed.append((x, y))


class _EKFWithCovariance:
    def __init__(self, diag):
        self._covariance = np.diag(diag)

    def get_covariance(self):
        return self._covariance.copy()

    def max_variance(self):
        return float(np.max(np.diag(self._covariance)))


def test_discover_marks_collapsed_snapped_frontier_traversed(monkeypatch, caplog):
    planner = _PlannerWithCollapsedSnap()
    ctx = state_machine.RobotContext()
    ctx.planner = planner
    ctx.picam2 = object()
    ctx.color_override = "blue"
    ctx.x = 143.4
    ctx.y = 77.7
    ctx.run_start = state_machine.time.monotonic()

    monkeypatch.setattr(fsm_states, "_correction_due", lambda ctx: None)
    monkeypatch.setattr(fsm_states, "_sync", lambda ctx: None)
    monkeypatch.setattr(state_machine.vision, "capture_frame", lambda picam2: None)
    monkeypatch.setattr(state_machine.navigation, "get_pose", lambda: (143.4, 77.7, 278.7))

    with caplog.at_level("INFO"):
        result = state_machine._handle_discover(ctx)

    assert result == state_machine.State.DISCOVER
    assert planner.logged_traversed == [(142.5, 107.5)]
    assert ctx.current_path == []
    assert "A* adjusted frontier (142.5, 107.5) -> goal (142.5, 77.5) first None" in caplog.text
    assert "snapped to current cell via A*" in caplog.text


def test_handle_update_small_success_sets_wall_cooldown(monkeypatch, caplog):
    ctx = state_machine.RobotContext()
    ctx.ekf = _EKFWithCovariance([144.0, 9.0, 0.0025])
    ctx.return_state = state_machine.State.DISCOVER
    ctx.home_heading = 0.0
    ctx.run_start = state_machine.time.monotonic()

    pose = [97.1, 171.7, 359.6]

    monkeypatch.setattr(fsm_states, "_nearest_wall", lambda: ("north", 41.3))
    monkeypatch.setattr(state_machine.navigation, "get_pose", lambda: tuple(pose))
    monkeypatch.setattr(state_machine.navigation, "pivot_to_heading", lambda heading: None)

    def fake_correct_pose_from_wall(wall):
        assert wall == "north"
        pose[1] = 171.9
        pose[2] = 359.7
        return True

    monkeypatch.setattr(state_machine.navigation, "correct_pose_from_wall", fake_correct_pose_from_wall)

    with caplog.at_level("INFO"):
        result = state_machine._handle_update(ctx)

    assert result == state_machine.State.DISCOVER
    assert "diag(P)=[144.00, 9.00, 0.0025]" in caplog.text
    assert "Small correction" in caplog.text
    assert "north" in ctx._wall_update_cooldown
    assert ctx._wall_update_cooldown["north"] > state_machine.time.monotonic()


def test_correction_due_skips_wall_under_cooldown(monkeypatch):
    ctx = state_machine.RobotContext()
    ctx.ekf = _EKFWithCovariance([144.0, 9.0, 0.0025])
    ctx._wall_update_cooldown["north"] = state_machine.time.monotonic() + 5.0

    monkeypatch.setattr(fsm_states, "_nearest_wall", lambda: ("north", 41.0))

    assert state_machine._correction_due(ctx) is None


def test_handle_deliver_retries_when_navigation_fails(monkeypatch, caplog):
    ctx = state_machine.RobotContext()
    ctx.zone_x = 32.2
    ctx.zone_y = 37.4
    ctx.run_start = state_machine.time.monotonic()

    monkeypatch.setattr(fsm_states, "_correction_due", lambda ctx: None)
    monkeypatch.setattr(fsm_states, "_deliver_drive_to_zone", lambda ctx: "failed")
    monkeypatch.setattr(
        fsm_states,
        "_deliver_align_at_zone",
        lambda ctx: (_ for _ in ()).throw(AssertionError("align should not run")),
    )
    monkeypatch.setattr(
        fsm_states,
        "_deliver_bookkeeping",
        lambda ctx: (_ for _ in ()).throw(AssertionError("bookkeeping should not run")),
    )
    monkeypatch.setattr(
        fsm_states,
        "_deliver_async_image",
        lambda ctx: (_ for _ in ()).throw(AssertionError("image save should not run")),
    )
    monkeypatch.setattr(
        fsm_states,
        "_deliver_safe_backup",
        lambda ctx: (_ for _ in ()).throw(AssertionError("backup should not run")),
    )

    with caplog.at_level("INFO"):
        result = state_machine._handle_deliver(ctx)

    assert result == state_machine.State.DELIVER
    assert "[DELIVER] Arrived" not in caplog.text