import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import pytest
from fp.fsm import states as fsm_states
from fp.fsm.context import RobotContext, FSMTestHooks
from fp.fsm.states import (
    State,
    _STATE_CLASSES,
    InitState, DiscoverState, ApproachState, GrabState,
    DeliverState, AvoidState, UpdateState, CompleteState,
)


class TestStateRegistry:
    def test_all_non_done_states_have_class(self):
        for s in State:
            if s == State.DONE:
                continue
            assert s in _STATE_CLASSES, f"{s} missing from _STATE_CLASSES"

    def test_done_not_in_registry(self):
        assert State.DONE not in _STATE_CLASSES

    def test_each_entry_is_base_state_instance(self):
        for s, obj in _STATE_CLASSES.items():
            assert hasattr(obj, 'on_enter'), f"{s} class missing on_enter"
            assert hasattr(obj, 'step'),     f"{s} class missing step"
            assert hasattr(obj, 'on_exit'),  f"{s} class missing on_exit"


class TestLifecycleMethods:

    def _ctx(self):
        ctx = RobotContext()
        ctx.run_start = __import__('time').monotonic()
        ctx.return_state = State.DISCOVER
        return ctx

    @pytest.mark.parametrize("state_cls", [
        InitState, DiscoverState, ApproachState, GrabState,
        DeliverState, AvoidState, UpdateState, CompleteState,
    ])
    def test_on_enter_does_not_raise(self, state_cls):
        state_cls().on_enter(self._ctx())

    @pytest.mark.parametrize("state_cls", [
        InitState, DiscoverState, ApproachState, GrabState,
        DeliverState, AvoidState, UpdateState, CompleteState,
    ])
    def test_on_exit_does_not_raise(self, state_cls):
        state_cls().on_exit(self._ctx())


class TestDiscoverStateStep:
    def test_delegates_to_handle_discover(self, monkeypatch):
        ctx = RobotContext()
        ctx.run_start = __import__('time').monotonic()
        ctx.picam2 = object()
        ctx.color_override = "blue"

        calls = []

        def fake_handle_discover(c):
            calls.append(c)
            return State.DISCOVER

        monkeypatch.setattr(fsm_states, "_handle_discover", fake_handle_discover)

        result = DiscoverState().step(ctx)

        assert calls == [ctx]
        assert result == State.DISCOVER

    def test_step_returns_complete_on_time_expired(self, monkeypatch):
        ctx = RobotContext()
        ctx.run_start = 0.0
        ctx.run_limit_s = 0.0  # already expired

        monkeypatch.setattr(fsm_states, "_correction_due", lambda c: None)
        monkeypatch.setattr(fsm_states.vision, "capture_frame", lambda p: None)
        monkeypatch.setattr(fsm_states.navigation, "get_pose", lambda: (50.0, 50.0, 0.0))

        result = DiscoverState().step(ctx)
        assert result == State.COMPLETE


class TestInitStateStep:
    def test_delegates_to_handle_init(self, monkeypatch):
        ctx = RobotContext()
        ctx.hooks = FSMTestHooks(skip_init_calibration=True)
        calls = []

        def fake_handle_init(c):
            calls.append(c)
            c.run_start = __import__('time').monotonic()
            return State.DISCOVER

        monkeypatch.setattr(fsm_states, "_handle_init", fake_handle_init)
        monkeypatch.setattr(fsm_states.hal, "gripper_open", lambda: None)

        result = InitState().step(ctx)

        assert calls == [ctx]
        assert result == State.DISCOVER


class TestDeliverStateStep:
    def test_returns_deliver_on_failed_drive(self, monkeypatch):
        ctx = RobotContext()
        ctx.zone_x = 32.2
        ctx.zone_y = 37.4
        ctx.run_start = __import__('time').monotonic()

        monkeypatch.setattr(fsm_states, "_correction_due", lambda c: None)
        monkeypatch.setattr(fsm_states, "_deliver_drive_to_zone", lambda c: "failed")
        monkeypatch.setattr(fsm_states, "_deliver_align_at_zone",
                            lambda c: (_ for _ in ()).throw(AssertionError("should not run")))

        result = DeliverState().step(ctx)
        assert result == State.DELIVER

    def test_returns_complete_on_expired_drive(self, monkeypatch):
        ctx = RobotContext()
        ctx.run_start = __import__('time').monotonic()

        monkeypatch.setattr(fsm_states, "_correction_due", lambda c: None)
        monkeypatch.setattr(fsm_states, "_deliver_drive_to_zone", lambda c: "expired")

        result = DeliverState().step(ctx)
        assert result == State.COMPLETE


class TestCompleteStateStep:
    def test_returns_done(self, monkeypatch):
        ctx = RobotContext()
        monkeypatch.setattr(fsm_states.hal, "gripper_open", lambda: None)
        monkeypatch.setattr(fsm_states.navigation, "get_pose", lambda: (0.0, 0.0, 0.0))

        result = CompleteState().step(ctx)
        assert result == State.DONE


class TestUpdateStateStep:
    def test_delegates_to_handle_update(self, monkeypatch):
        ctx = RobotContext()
        ctx.run_start = __import__('time').monotonic()
        ctx.return_state = State.DISCOVER

        calls = []

        def fake_handle_update(c):
            calls.append(c)
            return State.DISCOVER

        monkeypatch.setattr(fsm_states, "_handle_update", fake_handle_update)

        result = UpdateState().step(ctx)
        assert calls == [ctx]
        assert result == State.DISCOVER
