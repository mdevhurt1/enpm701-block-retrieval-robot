import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub


class _FakeEKF:
    def __init__(self, pose):
        self._pose = pose
        self.heading_updates = []
        self.wall_updates = []

    def get_pose(self):
        return self._pose

    def update_heading(self, heading_deg, R_override=None):
        self.heading_updates.append((heading_deg, R_override))
        return True

    def update_wall(self, wall, dist_cm, R_override=None):
        self.wall_updates.append((wall, dist_cm, R_override))
        return True


class TestUltrasonicDiagnostics:
    def test_wall_too_close_ignored_while_carrying(self, monkeypatch, caplog):
        from fp import hal

        bot = hal.Robot()
        bot.set_carrying_block(True)
        monkeypatch.setattr(
            bot,
            "_ping",
            lambda trig_pin, echo_pin, timeout_s=0.03: 7.0,
        )

        with caplog.at_level(logging.WARNING):
            assert bot.wall_too_close() is False

        assert "wall_too_close triggered" not in caplog.text

    def test_wall_too_close_logs_recent_short_readings(self, monkeypatch, caplog):
        from fp import hal

        bot = hal.Robot()
        readings = iter([55.0, 41.2, 12.3])
        monkeypatch.setattr(
            bot,
            "_ping",
            lambda trig_pin, echo_pin, timeout_s=0.03: next(readings),
        )

        assert bot.wall_too_close() is False
        assert bot.wall_too_close() is False

        with caplog.at_level(logging.WARNING):
            assert bot.wall_too_close() is True

        assert "[ULTRA] wall_too_close triggered by 12.3 cm reading" in caplog.text
        assert "recent front readings=[55.0, 41.2, 12.3]" in caplog.text

    def test_wall_too_close_logs_recent_timeouts(self, monkeypatch, caplog):
        from fp import hal

        bot = hal.Robot()
        readings = iter([50.0, None, None])
        monkeypatch.setattr(
            bot,
            "_ping",
            lambda trig_pin, echo_pin, timeout_s=0.03: next(readings),
        )

        assert bot.wall_too_close() is False
        assert bot.wall_too_close() is False

        with caplog.at_level(logging.WARNING):
            assert bot.wall_too_close() is True

        assert "consecutive timeouts" in caplog.text
        assert "recent front readings=[50.0, None, None]" in caplog.text


class TestRuntimeWallCorrection:
    def test_correct_pose_from_wall_runs_sweep_on_large_residual(self, monkeypatch, caplog):
        from fp import navigation, localization

        fake_ekf = _FakeEKF((47.6, 146.7, 270.0))
        monkeypatch.setattr(localization.loc, "ekf", fake_ekf)
        monkeypatch.setattr(localization.loc, "odo_path", [])
        monkeypatch.setattr(localization, "_trimmed_mean_distance", lambda n=5: 60.0)
        monkeypatch.setattr(localization.hal, "get_heading", lambda: 100.0)

        sweep_calls = []

        def fake_sweep(center_imu, sweep_half_deg=25.0, n_samples=11, log_prefix="[INIT]"):
            sweep_calls.append((center_imu, sweep_half_deg, n_samples, log_prefix))
            return (88.7, 33.0)

        monkeypatch.setattr(localization, "_sweep_and_fit_perpendicular", fake_sweep)

        with caplog.at_level(logging.INFO):
            accepted = navigation.correct_pose_from_wall("west")

        assert accepted is True
        assert sweep_calls == [(100.0, 15.0, 7, "[GTC]")]
        assert fake_ekf.heading_updates[0][0] == 270.0
        assert fake_ekf.wall_updates[0][0:2] == ("west", 33.0)
        assert "expected=32.6 cm" in caplog.text
        assert "running wall sweep" in caplog.text
        assert "Sweep-refined wall distance: measured=33.0 cm" in caplog.text

    def test_correct_pose_from_wall_skips_sweep_on_small_residual(self, monkeypatch, caplog):
        from fp import navigation, localization

        fake_ekf = _FakeEKF((50.0, 120.0, 270.0))
        monkeypatch.setattr(localization.loc, "ekf", fake_ekf)
        monkeypatch.setattr(localization.loc, "odo_path", [])
        monkeypatch.setattr(localization, "_trimmed_mean_distance", lambda n=5: 36.0)
        monkeypatch.setattr(localization.hal, "get_heading", lambda: 95.0)

        def should_not_run(*args, **kwargs):
            raise AssertionError("runtime sweep should not run for a small residual")

        monkeypatch.setattr(localization, "_sweep_and_fit_perpendicular", should_not_run)

        with caplog.at_level(logging.INFO):
            accepted = navigation.correct_pose_from_wall("west")

        assert accepted is True
        assert fake_ekf.heading_updates[0][0] == 270.0
        assert fake_ekf.wall_updates[0][0:2] == ("west", 36.0)
        assert "expected=35.0 cm" in caplog.text
        assert "running wall sweep" not in caplog.text


class TestPathExecution:
    def test_follow_path_aborts_on_failed_waypoint(self, monkeypatch, caplog):
        from fp import navigation, motion, localization

        class _Planner:
            def __init__(self):
                self.logged = []

            def log_traversed(self, x, y):
                self.logged.append((x, y))

        planner = _Planner()
        pose = [0.0, 0.0, 0.0]

        monkeypatch.setattr(localization, "get_pose", lambda: tuple(pose))

        def fake_navigate_to(wx, wy, cx, cy, correct_on_arrival=True):
            pose[0] = 4.0
            return pose[0], pose[1], 'wall'

        monkeypatch.setattr(motion, "navigate_to", fake_navigate_to)

        with caplog.at_level(logging.WARNING):
            result = navigation.follow_path([(20.0, 0.0)], planner)

        assert result is False
        assert planner.logged == [(4.0, 0.0)]
        assert "follow_path: waypoint (20.0, 0.0) failed with wall" in caplog.text


class TestCalibrationDistanceAggregation:
    def test_batched_trimmed_mean_distance_discards_outlier_batches(self, monkeypatch, caplog):
        from fp import localization
        import time as _time

        batch_estimates = iter([70.0, 72.0, 71.0, 120.0, 69.0])
        monkeypatch.setattr(localization, "_trimmed_mean_distance", lambda n=5: next(batch_estimates))
        monkeypatch.setattr(_time, "sleep", lambda seconds: None)

        with caplog.at_level(logging.INFO):
            result = localization._batched_trimmed_mean_distance(
                n_batches=5,
                samples_per_batch=5,
                log_prefix="[INIT]",
                label="North wall consistency",
            )

        assert result == 71.0
        assert "North wall consistency batches: [69.0, 70.0, 71.0, 72.0, 120.0] -> robust=71.0 cm" in caplog.text