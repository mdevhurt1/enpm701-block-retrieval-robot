import time
import pytest
from unittest.mock import patch, MagicMock
from fp import fsm, arena, vision
from fp.fsm import FSM, State
from fp.localization import Localizer
from fp.motion import TargetDetected, ApproachVisualStuck


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


def test_localize_after_grab_runs_pre_carry_localize_with_bypass_cooldown():
    # When carrying a block into the LOCALIZE state, the FSM must run an
    # extra localize call with bypass_cooldown=True BEFORE the move home.
    # Without this the long carry leg drives blindly with whatever odometry
    # the approach finished with - the SE->NW carry observed landing at
    # the centre of the north wall is the failure mode this guards.
    f = _make_fsm()
    f.holding_target = True
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize") as mock_localize:
        f._on_localize()

    bypass_calls = [c for c in mock_localize.call_args_list
                    if c.kwargs.get("bypass_cooldown") is True]
    assert len(bypass_calls) == 2


def test_localize_without_grab_does_not_run_pre_carry_localize():
    # Symmetric guard: when not carrying a block, no bypass_cooldown call.
    # A pre-carry localize on every LOCALIZE entry would burn ~10 s per
    # cell on the search loop and undo the cooldown gating we have for
    # the search/approach phases.
    f = _make_fsm()
    f.holding_target = False
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize") as mock_localize:
        f._on_localize()

    bypass_calls = [c for c in mock_localize.call_args_list
                    if c.kwargs.get("bypass_cooldown") is True]
    assert len(bypass_calls) == 0


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


def test_search_marks_visited_when_move_to_point_fails():
    # When move_to_point can't reach the target waypoint, the FSM must
    # mark that cell visited so the lawnmower doesn't re-queue it immediately.
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point", return_value=False), \
         patch.object(f.planner, "scan_360", return_value=None), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch.object(f.planner, "mark_visited") as mock_mark:
        f._on_search()
    mock_mark.assert_any_call(100.0, 100.0)


def test_search_uses_cached_sighting_when_available():
    # When the FSM searches for a color it already saw as a non-target on a
    # previous scan, head straight to that recorded position instead of
    # grinding through the lawnmower grid.
    f = _make_fsm()
    target = f._target_color()
    f.planner._record_sighting(target, 150.0, 75.0, 0.0)

    with patch("fp.fsm.motion.move_to_point") as mock_move, \
         patch.object(f.planner, "next_pose") as mock_next_pose, \
         patch.object(f.planner, "scan_360", return_value=None):
        next_state = f._on_search()

    assert next_state == State.SEARCH
    mock_next_pose.assert_not_called()
    args = mock_move.call_args[0]
    assert args[1] == pytest.approx(150.0)
    assert args[2] == pytest.approx(75.0)


def test_search_falls_back_to_next_pose_when_no_sighting():
    # Empty sightings cache -> next_pose lawnmower behavior preserved.
    f = _make_fsm()

    with patch("fp.fsm.motion.move_to_point") as mock_move, \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)) as mock_next_pose, \
         patch.object(f.planner, "scan_360", return_value=None):
        next_state = f._on_search()

    assert next_state == State.SEARCH
    mock_next_pose.assert_called_once()
    args = mock_move.call_args[0]
    assert args[1] == pytest.approx(100.0)
    assert args[2] == pytest.approx(100.0)


def test_search_consumes_cached_sighting():
    # get_sighting pops, so a stale sighting that didn't pan out won't loop
    # us forever - second call should fall through to next_pose.
    f = _make_fsm()
    target = f._target_color()
    f.planner._record_sighting(target, 150.0, 75.0, 0.0)

    with patch("fp.fsm.motion.move_to_point"), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)) as mock_next_pose, \
         patch.object(f.planner, "scan_360", return_value=None):
        f._on_search()
        f._on_search()

    # First call consumes the sighting; second call hits next_pose.
    mock_next_pose.assert_called_once()


def test_search_skips_scan_when_quality_below_threshold():
    # Cell is already-quality-searched -> skip the ~22 s scan_360 entirely.
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point"), \
         patch.object(f.planner, "get_search_quality", return_value=0.0), \
         patch.object(f.planner, "has_sighting_in_cell", return_value=False), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch.object(f.planner, "scan_360") as mock_scan:
        next_state = f._on_search()

    mock_scan.assert_not_called()
    assert next_state == State.SEARCH


def test_search_runs_scan_when_low_quality_but_cell_has_sighting():
    # Sighting overrides skip-on-low-quality: a cached sighting is intel
    # that a block was visible from here, and the colour we previously
    # logged as non-target may now be our target. Re-scan.
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point"), \
         patch.object(f.planner, "get_search_quality", return_value=0.0), \
         patch.object(f.planner, "has_sighting_in_cell", return_value=True), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch.object(f.planner, "scan_360", return_value=None) as mock_scan:
        f._on_search()

    mock_scan.assert_called_once()


def test_search_runs_scan_when_quality_high_and_no_sighting():
    # Fresh cell, no cached sighting -> normal scan_360 path.
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point"), \
         patch.object(f.planner, "get_search_quality", return_value=100.0), \
         patch.object(f.planner, "has_sighting_in_cell", return_value=False), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch.object(f.planner, "scan_360", return_value=None) as mock_scan:
        f._on_search()

    mock_scan.assert_called_once()


def test_search_skip_marks_visited():
    # Skipping the scan still has to register the visit, otherwise the
    # cell never accumulates visits and the planner's next_pose biasing
    # against revisiting it stops working.
    f = _make_fsm()
    with patch("fp.fsm.motion.move_to_point"), \
         patch.object(f.planner, "get_search_quality", return_value=0.0), \
         patch.object(f.planner, "has_sighting_in_cell", return_value=False), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch.object(f.planner, "mark_visited") as mock_mark, \
         patch.object(f.planner, "scan_360", return_value=None):
        f._on_search()

    mock_mark.assert_called()


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


def test_step_out_transitions_to_search():
    f = _make_fsm()
    with patch("fp.fsm.motion.move_reverse"), \
         patch("fp.fsm.motion.pivot_to_heading"):
        next_state = f._on_step_out()
    assert next_state == State.SEARCH


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
         patch("fp.fsm.vision.capture_frame"), \
         patch.object(f.planner, "scan_360", return_value=det), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)), \
         patch("fp.fsm.arena.in_collection_zone", return_value=True):
        state, delivered = f.run()

    assert state == State.COMPLETE
    assert delivered == 1


# -- Exception paths ------------------------------------------------------------

def test_localize_target_detected_transitions_to_approach():
    f = _make_fsm()
    det = vision.DetectionResult(label="red", cx=480, cy=360, area=5000)
    with patch("fp.fsm.motion.move_to_point", side_effect=TargetDetected(det)), \
         patch("fp.fsm.motion.localize"):
        next_state = f._on_localize()
    assert next_state == State.APPROACH


def test_search_target_detected_transitions_to_approach():
    f = _make_fsm()
    det = vision.DetectionResult(label="red", cx=480, cy=360, area=5000)
    with patch("fp.fsm.motion.move_to_point", side_effect=TargetDetected(det)), \
         patch.object(f.planner, "scan_360", return_value=None), \
         patch.object(f.planner, "next_pose", return_value=(100.0, 100.0)):
        next_state = f._on_search()
    assert next_state == State.APPROACH


def test_approach_target_detected_transitions_to_grab():
    f = _make_fsm()
    det = vision.DetectionResult(label="red", cx=480, cy=360, area=5000)
    with patch("fp.fsm.motion.approach_visual", side_effect=TargetDetected(det)):
        next_state = f._on_approach()
    assert next_state == State.GRAB


def test_approach_visual_stuck_transitions_to_localize():
    f = _make_fsm()
    with patch("fp.fsm.motion.approach_visual", side_effect=ApproachVisualStuck()):
        next_state = f._on_approach()
    assert next_state == State.LOCALIZE


# -- step() no-handler branch ---------------------------------------------------

def test_step_no_handler_returns_false():
    f = _make_fsm()
    f.state = State.COMPLETE   # not in _handlers dict
    f.delivered = 0             # prevent the delivered>=max_deliveries early exit
    assert f.step() is False


# -- module-level run() ---------------------------------------------------------

def test_module_level_run_returns_complete_after_one_delivery():
    loc = Localizer()
    loc.set_position(*arena.LAUNCH_ZONE_CENTER)
    det = vision.DetectionResult(label="red", cx=480, cy=360,
                                 area=vision.IN_REACH_TOLERANCE["red"] + 1)
    with patch("fp.fsm.motion.move_to_point"), \
         patch("fp.fsm.motion.localize"), \
         patch("fp.fsm.motion.approach_visual", return_value=True), \
         patch("fp.fsm.motion.grab_block", return_value=True), \
         patch("fp.fsm.motion.release_block"), \
         patch("fp.fsm.motion.move_reverse"), \
         patch("fp.fsm.motion.pivot_to_heading"), \
         patch("fp.fsm.vision.capture_frame"), \
         patch("fp.search.SearchPlanner.scan_360", return_value=det), \
         patch("fp.search.SearchPlanner.next_pose", return_value=(100.0, 100.0)), \
         patch("fp.fsm.arena.in_collection_zone", return_value=True):
        state, delivered = fsm.run(loc, max_deliveries=1)
    assert state == State.COMPLETE
    assert delivered == 1
