import os
import pytest
from fp.mapping import PathTracker
from unittest.mock import patch, MagicMock
from fp.localization import Localizer
from fp import motion


def test_no_segments_initially():
    t = PathTracker()
    assert t._segments == []


def test_start_segment_adds_a_segment():
    t = PathTracker()
    t.start_segment()
    assert len(t._segments) == 1


def test_start_segment_uses_different_colors():
    t = PathTracker()
    t.start_segment()
    t.start_segment()
    color_a = t._segments[0][0]
    color_b = t._segments[1][0]
    for c in (color_a, color_b):
        assert len(c) == 3
        assert all(0.0 <= v <= 1.0 for v in c)
    assert color_a != color_b


def test_record_appends_to_current_segment():
    t = PathTracker()
    t.start_segment()
    t.record(10.0, 20.0)
    t.record(30.0, 40.0)
    assert t._segments[0][1] == [(10.0, 20.0), (30.0, 40.0)]


def test_record_before_any_segment_is_a_no_op():
    t = PathTracker()
    t.record(5.0, 5.0)
    assert t._segments == []


def test_record_goes_to_latest_segment():
    t = PathTracker()
    t.start_segment()
    t.record(1.0, 2.0)
    t.start_segment()
    t.record(3.0, 4.0)
    assert t._segments[0][1] == [(1.0, 2.0)]
    assert t._segments[1][1] == [(3.0, 4.0)]


def test_save_plot_creates_file(tmp_path):
    t = PathTracker()
    t.start_segment()
    t.record(30.0, 30.0)
    t.record(60.0, 60.0)
    t.start_segment()
    t.record(90.0, 90.0)
    out = str(tmp_path / "path.png")
    t.save_plot(out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0


def test_save_plot_with_no_segments_does_not_raise(tmp_path):
    t = PathTracker()
    out = str(tmp_path / "empty.png")
    t.save_plot(out)
    assert os.path.isfile(out)


def test_save_plot_skips_single_point_segments(tmp_path):
    t = PathTracker()
    t.start_segment()
    t.record(50.0, 50.0)
    out = str(tmp_path / "single.png")
    t.save_plot(out)
    assert os.path.isfile(out)


def test_localizer_tracker_is_none_by_default():
    loc = Localizer()
    assert loc.tracker is None


def test_apply_forward_records_when_tracker_attached():
    loc = Localizer()
    tracker = PathTracker()
    tracker.start_segment()
    loc.tracker = tracker
    loc.set_position(0.0, 0.0)

    loc.apply_forward(10.0, 0.0)

    assert len(tracker._segments[0][1]) == 1
    x, y = tracker._segments[0][1][0]
    assert y == pytest.approx(10.0, abs=1e-6)
    assert x == pytest.approx(0.0, abs=1e-6)


def test_apply_forward_does_not_record_without_tracker():
    loc = Localizer()
    loc.set_position(0.0, 0.0)
    loc.apply_forward(10.0, 0.0)
    assert loc.pose.y_cm == pytest.approx(10.0)


def test_apply_reverse_records_via_apply_forward():
    loc = Localizer()
    tracker = PathTracker()
    tracker.start_segment()
    loc.tracker = tracker
    loc.set_position(0.0, 50.0)

    loc.apply_reverse(10.0, 0.0)

    assert len(tracker._segments[0][1]) == 1
    _, y = tracker._segments[0][1][0]
    assert y == pytest.approx(40.0, abs=1e-6)


def test_drive_calls_start_segment_when_tracker_attached():
    loc = Localizer()
    loc.set_position(50.0, 50.0)
    mock_tracker = MagicMock()
    loc.tracker = mock_tracker

    with patch("fp.motion.hal.set_individual_duties"), \
         patch("fp.motion.hal.stop_motors"), \
         patch("fp.motion.hal.get_heading", return_value=0.0), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=200.0), \
         patch("fp.motion.hal.PWM_DUTY_STRAIGHT", 50), \
         patch("fp.motion.hal.COUNTS_PER_CM", 10), \
         patch("fp.motion.hal.read_front_left_encoder", side_effect=[0] + [500] * 30), \
         patch("fp.motion.hal.read_back_right_encoder", side_effect=[0] + [500] * 30), \
         patch("fp.motion.time.monotonic", return_value=100.0), \
         patch("fp.motion.vision.detect_all_colors", return_value={}):
        motion._drive(loc, 10.0, 0.0, +1)

    mock_tracker.start_segment.assert_called_once()


def test_drive_does_not_call_start_segment_without_tracker():
    loc = Localizer()
    loc.set_position(50.0, 50.0)

    with patch("fp.motion.hal.set_individual_duties"), \
         patch("fp.motion.hal.stop_motors"), \
         patch("fp.motion.hal.get_heading", return_value=0.0), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=200.0), \
         patch("fp.motion.hal.PWM_DUTY_STRAIGHT", 50), \
         patch("fp.motion.hal.COUNTS_PER_CM", 10), \
         patch("fp.motion.hal.read_front_left_encoder", side_effect=[0] + [500] * 30), \
         patch("fp.motion.hal.read_back_right_encoder", side_effect=[0] + [500] * 30), \
         patch("fp.motion.time.monotonic", return_value=100.0), \
         patch("fp.motion.vision.detect_all_colors", return_value={}):
        motion._drive(loc, 10.0, 0.0, +1)
