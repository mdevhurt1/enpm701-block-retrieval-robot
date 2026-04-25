import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import queue
import threading

import pytest

from fp import tuning
from fp.fsm.runner import FsmControl
from fp.fsm.states import State
from fp.fsm.context import RobotContext
from fp.repl.commands import CommandEnv, dispatch, _COMMANDS
from fp.repl.tuning_diff import (
    capture_defaults,
    compute_diff,
    load_from_toml,
    load_toml_if_exists,
    save_to_toml,
)


# Helpers

def _make_env(tmp_path, snapshot=None):
    ctx = RobotContext()
    control = FsmControl()
    if snapshot is None:
        snapshot = capture_defaults()
    return CommandEnv(
        ctx=ctx,
        control=control,
        session_snapshot=snapshot,
        toml_path=str(tmp_path / "runtime.toml"),
    )


# Command registry

class TestCommandRegistry:
    def test_all_expected_commands_registered(self):
        expected = {
            "pose", "cov", "state", "blocks",
            "pause", "resume", "step", "force",
            "tune", "set", "revert",
            ":diff", ":save", ":reset",
            "help",
        }
        for name in expected:
            assert name in _COMMANDS, f"Command {name!r} not registered"

    def test_each_entry_is_callable(self):
        for name, (handler, desc) in _COMMANDS.items():
            assert callable(handler), f"{name}: handler not callable"
            assert isinstance(desc, str), f"{name}: desc not a string"


# dispatch() parsing

class TestDispatch:
    def test_empty_line_returns_empty(self, tmp_path):
        env = _make_env(tmp_path)
        assert dispatch("", env) == ""

    def test_whitespace_only_returns_empty(self, tmp_path):
        env = _make_env(tmp_path)
        assert dispatch("   ", env) == ""

    def test_unknown_command_returns_error_hint(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("xyzzy", env)
        assert "Unknown command" in result
        assert "help" in result.lower()

    def test_case_insensitive_dispatch(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("HELP", env)
        assert "Available commands" in result

    def test_dispatch_passes_args(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path)
        received = []
        monkeypatch.setitem(_COMMANDS, "testcmd", (
            lambda args, e: received.append(args) or "ok",
            "test",
        ))
        dispatch("testcmd arg1 arg2", env)
        assert received == [["arg1", "arg2"]]


# Inspection commands

class TestInspectionCommands:
    def test_pose_calls_navigation(self, tmp_path, monkeypatch):
        import fp.navigation as nav
        monkeypatch.setattr(nav, "get_pose", lambda: (12.3, 45.6, 270.0))
        env = _make_env(tmp_path)
        result = dispatch("pose", env)
        assert "12.3" in result
        assert "45.6" in result
        assert "270.0" in result

    def test_state_when_none(self, tmp_path):
        env = _make_env(tmp_path)
        env.control.current_state = None
        result = dispatch("state", env)
        assert "not started" in result

    def test_state_when_set(self, tmp_path):
        env = _make_env(tmp_path)
        env.control.current_state = State.DISCOVER
        result = dispatch("state", env)
        assert "DISCOVER" in result

    def test_blocks_shows_scoring(self, tmp_path):
        env = _make_env(tmp_path)
        env.ctx.blocks_delivered = 3
        env.ctx.score = 30
        env.ctx.color_override = "red"
        result = dispatch("blocks", env)
        assert "3" in result
        assert "30" in result
        assert "red" in result


# Control commands

class TestControlCommands:
    def test_pause_sets_pause_event(self, tmp_path):
        env = _make_env(tmp_path)
        assert not env.control.pause_event.is_set()
        dispatch("pause", env)
        assert env.control.pause_event.is_set()

    def test_pause_when_already_paused(self, tmp_path):
        env = _make_env(tmp_path)
        env.control.paused.set()
        result = dispatch("pause", env)
        assert "already paused" in result.lower()

    def test_resume_when_not_paused(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("resume", env)
        assert "not paused" in result.lower()

    def test_resume_sets_resume_event(self, tmp_path):
        env = _make_env(tmp_path)
        env.control.paused.set()
        dispatch("resume", env)
        assert env.control.resume_event.is_set()

    def test_step_when_not_paused(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("step", env)
        assert "not paused" in result.lower()

    def test_step_sets_step_event(self, tmp_path):
        env = _make_env(tmp_path)
        env.control.paused.set()
        dispatch("step", env)
        assert env.control.step_event.is_set()

    def test_force_unknown_state(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("force FLYING", env)
        assert "Unknown state" in result

    def test_force_done_rejected(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("force DONE", env)
        assert "Cannot" in result

    def test_force_queues_transition(self, tmp_path):
        env = _make_env(tmp_path)
        dispatch("force DISCOVER", env)
        assert not env.control.force_transition_queue.empty()
        assert env.control.force_transition_queue.get() == State.DISCOVER

    def test_force_no_arg(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("force", env)
        assert "Usage" in result


# Tuning commands

class TestTuningCommands:
    def setup_method(self):
        self._orig = tuning.get_param("ARRIVAL_TOLERANCE_CM")

    def teardown_method(self):
        tuning.set_param("ARRIVAL_TOLERANCE_CM", self._orig)

    def test_set_updates_param(self, tmp_path):
        env = _make_env(tmp_path)
        dispatch("set ARRIVAL_TOLERANCE_CM 12.0", env)
        assert tuning.get_param("ARRIVAL_TOLERANCE_CM") == pytest.approx(12.0)

    def test_set_int_param(self, tmp_path):
        env = _make_env(tmp_path)
        orig = tuning.get_param("WALL_SWEEP_SAMPLES")
        try:
            dispatch("set WALL_SWEEP_SAMPLES 11", env)
            assert tuning.get_param("WALL_SWEEP_SAMPLES") == 11
        finally:
            tuning.set_param("WALL_SWEEP_SAMPLES", orig)

    def test_set_unknown_param(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("set NONEXISTENT_PARAM 5.0", env)
        assert "Unknown param" in result

    def test_set_invalid_value(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("set ARRIVAL_TOLERANCE_CM notanumber", env)
        assert "Invalid" in result

    def test_set_no_args(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("set", env)
        assert "Usage" in result

    def test_revert_restores_value(self, tmp_path):
        snap = capture_defaults()
        snap["ARRIVAL_TOLERANCE_CM"] = 8.0
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 15.0)
        env = _make_env(tmp_path, snapshot=snap)
        dispatch("revert ARRIVAL_TOLERANCE_CM", env)
        assert tuning.get_param("ARRIVAL_TOLERANCE_CM") == pytest.approx(8.0)

    def test_revert_unknown_param(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("revert BOGUS", env)
        assert "Unknown param" in result

    def test_diff_empty_when_no_changes(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch(":diff", env)
        assert "No changes" in result

    def test_diff_shows_change(self, tmp_path):
        snap = capture_defaults()
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 20.0)
        env = _make_env(tmp_path, snapshot=snap)
        try:
            result = dispatch(":diff", env)
            assert "ARRIVAL_TOLERANCE_CM" in result
            assert "20.0" in result
        finally:
            tuning.set_param("ARRIVAL_TOLERANCE_CM", self._orig)

    def test_reset_reverts_all(self, tmp_path):
        snap = capture_defaults()
        snap["ARRIVAL_TOLERANCE_CM"] = 8.0
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 25.0)
        env = _make_env(tmp_path, snapshot=snap)
        dispatch(":reset", env)
        assert tuning.get_param("ARRIVAL_TOLERANCE_CM") == pytest.approx(8.0)

    def test_tune_lists_params(self, tmp_path):
        env = _make_env(tmp_path)
        result = dispatch("tune", env)
        assert "ARRIVAL_TOLERANCE_CM" in result
        assert "Tuning params" in result

    def test_tune_marks_changed(self, tmp_path):
        snap = capture_defaults()
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 99.0)
        env = _make_env(tmp_path, snapshot=snap)
        try:
            result = dispatch("tune", env)
            assert "* ARRIVAL_TOLERANCE_CM" in result or "*  ARRIVAL_TOLERANCE_CM" in result
        finally:
            tuning.set_param("ARRIVAL_TOLERANCE_CM", self._orig)


# tuning_diff persistence

class TestTuningDiff:
    def setup_method(self):
        self._orig = tuning.get_param("ARRIVAL_TOLERANCE_CM")

    def teardown_method(self):
        tuning.set_param("ARRIVAL_TOLERANCE_CM", self._orig)

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "runtime.toml")
        snap = capture_defaults()
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 17.5)
        n_saved = save_to_toml(path, snap)
        assert n_saved >= 1
        # Reset to default then reload
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 8.0)
        n_loaded = load_from_toml(path)
        assert n_loaded >= 1
        assert tuning.get_param("ARRIVAL_TOLERANCE_CM") == pytest.approx(17.5)

    def test_load_nonexistent_returns_zero(self, tmp_path):
        n = load_toml_if_exists(str(tmp_path / "missing.toml"))
        assert n == 0

    def test_save_merges_with_existing(self, tmp_path):
        path = str(tmp_path / "runtime.toml")
        # First session: save ARRIVAL_TOLERANCE_CM
        snap1 = capture_defaults()
        snap1["ARRIVAL_TOLERANCE_CM"] = 8.0
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 11.0)
        save_to_toml(path, snap1)
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 8.0)  # reset

        # Second session: save AVOID_BACKUP_CM only
        snap2 = capture_defaults()
        snap2["AVOID_BACKUP_CM"] = tuning.get_param("AVOID_BACKUP_CM")
        tuning.set_param("AVOID_BACKUP_CM", 20.0)
        save_to_toml(path, snap2)

        # Reload: both params should be present
        tuning.set_param("ARRIVAL_TOLERANCE_CM", 8.0)
        tuning.set_param("AVOID_BACKUP_CM", 15.0)
        load_from_toml(path)
        assert tuning.get_param("ARRIVAL_TOLERANCE_CM") == pytest.approx(11.0)
        assert tuning.get_param("AVOID_BACKUP_CM") == pytest.approx(20.0)
        # cleanup
        tuning.set_param("AVOID_BACKUP_CM", 15.0)

    def test_capture_defaults_is_snapshot(self):
        snap1 = capture_defaults()
        orig = tuning.get_param("ARRIVAL_TOLERANCE_CM")
        tuning.set_param("ARRIVAL_TOLERANCE_CM", orig + 5.0)
        snap2 = capture_defaults()
        assert snap1["ARRIVAL_TOLERANCE_CM"] == pytest.approx(orig)
        assert snap2["ARRIVAL_TOLERANCE_CM"] == pytest.approx(orig + 5.0)

    def test_compute_diff_empty_when_unchanged(self):
        snap = capture_defaults()
        assert compute_diff(snap) == {}

    def test_compute_diff_shows_change(self):
        snap = capture_defaults()
        orig = snap["ARRIVAL_TOLERANCE_CM"]
        tuning.set_param("ARRIVAL_TOLERANCE_CM", orig + 3.0)
        diff = compute_diff(snap)
        assert "ARRIVAL_TOLERANCE_CM" in diff
        assert diff["ARRIVAL_TOLERANCE_CM"] == (orig, pytest.approx(orig + 3.0))
