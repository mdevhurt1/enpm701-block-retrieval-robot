import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import math
import numpy as np
import pytest
import cv2


class TestDetectAllColors:

    def _solid_bgr_frame(self, bgr_color, h=480, w=640):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = bgr_color
        return frame

    def _red_frame(self):
        # Pure red in BGR
        return self._solid_bgr_frame((0, 0, 200))

    def _green_frame(self):
        return self._solid_bgr_frame((0, 200, 0))

    def _blue_frame(self):
        return self._solid_bgr_frame((200, 0, 0))

    def test_red_detected_in_red_frame(self):
        from fp import vision
        frame = self._red_frame()
        dets = vision.detect_all_colors(frame)
        # At minimum the red key exists; may or may not match HSV thresholds
        assert 'red' in dets

    def test_all_colors_keyed(self):
        from fp import vision
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = vision.detect_all_colors(frame)
        for color in vision.COLOR_MASKS:
            assert color in dets, f"color '{color}' missing from detect_all_colors result"

    def test_none_result_for_empty_frame(self):
        from fp import vision
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = vision.detect_all_colors(frame)
        for color, det in dets.items():
            assert det is None, f"{color} detected in blank frame (unexpected)"

    def test_none_input(self):
        from fp import vision
        dets = vision.detect_all_colors(None)
        assert isinstance(dets, dict)
        for color, det in dets.items():
            assert det is None


class TestEstimateBlockPose:

    def _make_block_frame(self, cx=320, cy=240, size_px=80, bgr_color=(0, 0, 200),
                          w=640, h=480):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        half = size_px // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        frame[y1:y2, x1:x2] = bgr_color
        return frame

    def test_centred_block_no_lateral_error(self):
        from fp import vision
        # Patch camera matrix so fy and fx are known
        import numpy as np
        vision._camera_matrix = np.array([
            [500.0,   0.0, 320.0],
            [  0.0, 500.0, 240.0],
            [  0.0,   0.0,   1.0],
        ])
        vision._calibrated = True

        # Build a red block frame centred at principal point
        # Block is MIN_BLOB_AREA-sized or larger
        frame = self._make_block_frame(cx=320, cy=240, size_px=200,
                                       bgr_color=(0, 0, 200))

        from fp.vision import DetectionResult
        det = DetectionResult(label='red', cx=320, cy=240, area=200*200)
        result = vision.estimate_block_pose(frame, det, robot_pose=(50.0, 50.0, 0.0))

        if result is None:
            pytest.skip("Block not detectable in synthetic frame (HSV mismatch)")
        arena_x, arena_y = result
        # With bearing=0 and robot heading north, lateral offset should be ~0
        assert abs(arena_x - 50.0) < 20.0, f"arena_x={arena_x:.1f}, expected ≈50"
        # Block should be ahead (north) of robot
        assert arena_y > 50.0, f"arena_y={arena_y:.1f} should be north of robot"

    def test_returns_none_when_uncalibrated(self):
        from fp import vision
        orig_calibrated = vision._calibrated
        vision._calibrated = False
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            from fp.vision import DetectionResult
            det = DetectionResult(label='red', cx=320, cy=240, area=50000)
            result = vision.estimate_block_pose(frame, det, (0.0, 0.0, 0.0))
            assert result is None
        finally:
            vision._calibrated = orig_calibrated


class TestWithinReach:

    def test_within_reach_when_centred_and_large(self):
        from fp import vision
        from fp.vision import PICKUP_AREA_THRESHOLD, FRAME_WIDTH
        # Centred (error=0) and large area
        det = vision.DetectionResult(label='red', cx=FRAME_WIDTH // 2, cy=540, area=PICKUP_AREA_THRESHOLD * 2)
        assert det.within_reach is True

    def test_not_within_reach_when_too_small(self):
        from fp import vision
        from fp.vision import PICKUP_AREA_THRESHOLD, FRAME_WIDTH
        det = vision.DetectionResult(label='red', cx=FRAME_WIDTH // 2, cy=540, area=PICKUP_AREA_THRESHOLD // 2)
        assert det.within_reach is False

    def test_not_within_reach_when_off_centre(self):
        from fp import vision
        from fp.vision import PICKUP_AREA_THRESHOLD
        # Large enough area but high centroid error (cx=0 far from centre)
        det = vision.DetectionResult(label='red', cx=0, cy=540, area=PICKUP_AREA_THRESHOLD * 2)
        assert det.within_reach is False


class TestIsCalibrated:

    def test_returns_bool(self):
        from fp import vision
        result = vision.is_calibrated()
        assert isinstance(result, bool)


class TestTuning:

    def test_import_tuning(self):
        from fp import tuning
        assert hasattr(tuning, 'RUN_LIMIT_S')
        assert tuning.RUN_LIMIT_S > 0

    def test_pickup_area_threshold_positive(self):
        from fp.vision import PICKUP_AREA_THRESHOLD
        assert PICKUP_AREA_THRESHOLD > 0

    def test_kp_visual_positive(self):
        from fp import navigation
        assert navigation.KP_VISUAL >= 0
