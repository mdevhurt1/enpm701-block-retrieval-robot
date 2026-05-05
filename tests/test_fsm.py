import time
import pytest
from unittest.mock import patch, MagicMock
from fp import fsm, arena, vision
from fp.fsm import FSM, State
from fp.localization import Localizer


def _make_fsm(x=arena.LAUNCH_ZONE_CENTER[0], y=arena.LAUNCH_ZONE_CENTER[1],
              max_deliveries=fsm.TOTAL_BLOCKS):
    loc = Localizer()
    loc.set_position(x, y)
    return FSM(loc, time.monotonic(), max_deliveries=max_deliveries)


def test_target_color_cycles_rgb():
    f = _make_fsm()
    f.delivered = 0
    assert f._target_color() == "red"
    f.delivered = 1
    assert f._target_color() == "green"
    f.delivered = 2
    assert f._target_color() == "blue"
    f.delivered = 3
    assert f._target_color() == "red"


def test_elapsed_s_returns_nonnegative():
    f = _make_fsm()
    elapsed = f._elapsed_s()
    assert elapsed >= 0.0
    assert isinstance(elapsed, float)


def test_step_returns_false_at_max_deliveries():
    f = _make_fsm(max_deliveries=3)
    f.delivered = 3
    assert f.step() is False
    assert f.state == State.COMPLETE


def test_step_returns_false_on_hard_timeout():
    loc = Localizer()
    # start_time far in the past so _elapsed_s() immediately exceeds T_HARD_S
    f = FSM(loc, time.monotonic() - fsm.T_HARD_S - 1.0)
    assert f.step() is False
    assert f.state == State.TIMEOUT


def test_localize_uninitialized_transitions_to_search():
    f = _make_fsm()
    assert f.state == State.LOCALIZE
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize"):
        next_state = f._on_localize()
    assert next_state == State.SEARCH


def test_localize_after_delivery_transitions_to_search():
    f = _make_fsm()
    f.holding_target = False
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize"):
        next_state = f._on_localize()
    assert next_state == State.SEARCH


def test_localize_after_grab_transitions_to_carry_home():
    f = _make_fsm()
    f.holding_target = True
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize"):
        next_state = f._on_localize()
    assert next_state == State.CARRY_HOME


def test_search_found_transitions_to_approach():
    f = _make_fsm()
    det = vision.DetectionResult(label="red", cx=480, cy=360, area=5000)
    with patch.object(f.planner, "scan_360", return_value=det):
        next_state = f._on_search()
    assert next_state == State.APPROACH


def test_search_not_found_stays_in_search():
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize"), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch.object(f.planner, "scan_360", return_value=None):
        next_state = f._on_search()
    assert next_state == State.SEARCH


def test_approach_success_transitions_to_grab():
    f = _make_fsm()
    with patch("fp.fsm.motion.approach_visual", return_value=True):
        next_state = f._on_approach()
    assert next_state == State.GRAB


def test_approach_failure_transitions_to_search():
    f = _make_fsm()
    with patch("fp.fsm.motion.approach_visual", return_value=False):
        next_state = f._on_approach()
    assert next_state == State.SEARCH


def test_grab_success_transitions_to_localize():
    f = _make_fsm()
    with patch("fp.fsm.motion.grab_block", return_value=True):
        next_state = f._on_grab()
    assert next_state == State.LOCALIZE
    assert f.holding_target is True


def test_grab_failure_transitions_to_search():
    f = _make_fsm()
    with patch("fp.fsm.motion.grab_block", return_value=False):
        next_state = f._on_grab()
    assert next_state == State.SEARCH


def test_carry_home_transitions_to_drop():
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point"):
        next_state = f._on_carry_home()
    assert next_state == State.DROP


def test_drop_in_zone_increments_and_steps_out():
    loc = Localizer()
    loc.set_position(*arena.COLLECTION_ZONE_CENTER)
    f = FSM(loc, time.monotonic())
    with patch("fp.fsm.motion.release_block"):
        next_state = f._on_drop()
    assert f.delivered == 1
    assert next_state == State.STEP_OUT


def test_drop_outside_zone_does_not_increment():
    loc = Localizer()
    loc.set_position(150.0, 50.0)
    f = FSM(loc, time.monotonic())
    with patch("fp.fsm.motion.release_block"):
        next_state = f._on_drop()
    assert f.delivered == 0
    assert next_state == State.STEP_OUT


def test_drop_transitions_to_complete_when_all_delivered():
    loc = Localizer()
    loc.set_position(*arena.COLLECTION_ZONE_CENTER)
    f = FSM(loc, time.monotonic(), max_deliveries=fsm.TOTAL_BLOCKS)
    f.delivered = fsm.TOTAL_BLOCKS - 1
    with patch("fp.fsm.motion.release_block"):
        next_state = f._on_drop()
    assert f.delivered == fsm.TOTAL_BLOCKS
    assert next_state == State.COMPLETE


def test_step_out_transitions_to_localize():
    f = _make_fsm()
    with patch("fp.fsm.motion.move_reverse"), \
         patch("fp.fsm.motion.pivot_to_heading"):
        next_state = f._on_step_out()
    assert next_state == State.LOCALIZE


def test_run_reaches_complete():
    loc = Localizer()
    loc.set_position(22.86, 22.86)
    det = vision.DetectionResult(label="red", cx=480, cy=360,
                                 area=vision.IN_REACH_TOLERANCE["red"] + 1)
    f = FSM(loc, time.monotonic(), max_deliveries=1)

    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize"), \
         patch("fp.fsm.motion.approach_visual", return_value=True), \
         patch("fp.fsm.motion.grab_block", return_value=True), \
         patch("fp.fsm.motion.release_block"), \
         patch("fp.fsm.motion.move_reverse"), \
         patch("fp.fsm.motion.pivot_to_heading"), \
         patch.object(f.planner, "scan_360", return_value=det), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch("fp.fsm.arena.in_collection_zone", return_value=True):
        state, delivered = f.run()

    assert state == State.COMPLETE
    assert delivered == 1
