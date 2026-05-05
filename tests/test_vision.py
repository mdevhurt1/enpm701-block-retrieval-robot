import numpy as np
import cv2
import pytest
from unittest.mock import patch
from fp import vision

# Override SENSOR_RESOLUTION so centroid_error is testable with known values.
# The blob-test frames are created at this size.
_SENSOR_H = 720
_SENSOR_W = 960
vision.SENSOR_RESOLUTION = (_SENSOR_H, _SENSOR_W)

# Fixed frame size for mask / contour / detect tests.
_TEST_H = _SENSOR_H
_TEST_W = _SENSOR_W

_CX, _CY = 480, 360
_R_BIG = 60
_R_SMALL = 10


def _hsv_frame_with_blob(h, s, v, cx, cy, radius):
    frame = np.zeros((_TEST_H, _TEST_W, 3), dtype=np.uint8)
    cv2.circle(frame, (cx, cy), radius, (h, s, v), -1)
    return frame


def _bgr_frame_with_blob(h, s, v, cx, cy, radius):
    hsv = _hsv_frame_with_blob(h, s, v, cx, cy, radius)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_within_reach_at_threshold():
    det = vision.DetectionResult(label="red", cx=_CX, cy=_CY, area=vision.IN_REACH_TOLERANCE["red"])
    assert det.within_reach is True


def test_within_reach_below_threshold():
    det = vision.DetectionResult(label="red", cx=_CX, cy=_CY,
                                 area=vision.IN_REACH_TOLERANCE["red"] - 1)
    assert det.within_reach is False


def test_block_too_close_at_threshold():
    det = vision.DetectionResult(label="red", cx=_CX, cy=_CY,
                                 area=vision.TOO_CLOSE_TOLERANCE["red"])
    assert det.block_too_close is True


def test_block_too_close_below_threshold():
    det = vision.DetectionResult(label="red", cx=_CX, cy=_CY,
                                 area=vision.TOO_CLOSE_TOLERANCE["red"] - 1)
    assert det.block_too_close is False


def test_centroid_error_at_center():
    # Formula: cx - SENSOR_RESOLUTION[0]//2 + SENSOR_RESOLUTION[0]//50
    # With SENSOR_RESOLUTION[0]=720: offset = 720//50 = 14.
    # cx at center -> 0 + 14 = 14.
    cx_center = vision.SENSOR_RESOLUTION[0] // 2
    det = vision.DetectionResult(label="red", cx=cx_center, cy=_CY, area=1000)
    assert det.centroid_error == 14


def test_centroid_error_right_of_center():
    cx_center = vision.SENSOR_RESOLUTION[0] // 2
    det = vision.DetectionResult(label="red", cx=cx_center + 100, cy=_CY, area=1000)
    assert det.centroid_error == 114


def test_centroid_error_left_of_center():
    cx_center = vision.SENSOR_RESOLUTION[0] // 2
    det = vision.DetectionResult(label="red", cx=cx_center - 100, cy=_CY, area=1000)
    assert det.centroid_error == -86


def test_build_mask_red_blob_hits():
    hsv = _hsv_frame_with_blob(5, 200, 200, _CX, _CY, _R_BIG)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["red"])
    assert mask[_CY, _CX] > 0


def test_build_mask_black_frame_empty():
    hsv = np.zeros((_TEST_H, _TEST_W, 3), dtype=np.uint8)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["red"])
    assert mask.sum() == 0


def test_build_mask_blue_blob_hits():
    hsv = _hsv_frame_with_blob(100, 200, 200, _CX, _CY, _R_BIG)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["blue"])
    assert mask[_CY, _CX] > 0


def test_best_contour_single_blob():
    hsv = _hsv_frame_with_blob(5, 200, 200, _CX, _CY, _R_BIG)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["red"])
    result = vision._best_contour(mask, "red")
    assert result is not None
    _, area, cx, cy = result
    assert area > vision.MIN_BLOB_AREA["red"]
    assert abs(cx - _CX) <= 5
    assert abs(cy - _CY) <= 5


def test_best_contour_below_threshold_returns_none():
    hsv = _hsv_frame_with_blob(5, 200, 200, _CX, _CY, _R_SMALL)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["red"])
    assert vision._best_contour(mask, "red") is None


def test_best_contour_empty_mask():
    mask = np.zeros((_TEST_H, _TEST_W), dtype=np.uint8)
    assert vision._best_contour(mask, "red") is None


def test_best_contour_prefers_larger_area():
    hsv = _hsv_frame_with_blob(5, 200, 200, 200, 100, _R_BIG)
    cv2.circle(hsv, (700, 400), _R_BIG + 30, (5, 200, 200), -1)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["red"])
    result = vision._best_contour(mask, "red")
    assert result is not None
    _, _, cx, _ = result
    assert cx > 500


def test_detect_block_none_frame():
    with patch("fp.vision.capture_frame", return_value=None):
        assert vision.detect_block("red") is None


def test_detect_block_black_frame():
    frame = np.zeros((_TEST_H, _TEST_W, 3), dtype=np.uint8)
    with patch("fp.vision.capture_frame", return_value=frame):
        assert vision.detect_block("red") is None


def test_detect_block_red_blob():
    frame = _bgr_frame_with_blob(5, 200, 200, _CX, _CY, _R_BIG)
    with patch("fp.vision.capture_frame", return_value=frame):
        result = vision.detect_block("red")
    assert result is not None
    assert result.label == "red"
    assert abs(result.cx - _CX) <= 5
    assert result.area > vision.MIN_BLOB_AREA["red"]


def test_detect_block_blue_blob():
    frame = _bgr_frame_with_blob(100, 200, 200, _CX, _CY, 100)
    with patch("fp.vision.capture_frame", return_value=frame):
        result = vision.detect_block("blue")
    assert result is not None
    assert result.label == "blue"


def test_detect_block_green_blob():
    frame = _bgr_frame_with_blob(55, 150, 180, _CX, _CY, _R_BIG)
    with patch("fp.vision.capture_frame", return_value=frame):
        result = vision.detect_block("green")
    assert result is not None
    assert result.label == "green"


def test_detect_block_wrong_color_returns_none():
    frame = _bgr_frame_with_blob(5, 200, 200, _CX, _CY, _R_BIG)
    with patch("fp.vision.capture_frame", return_value=frame):
        assert vision.detect_block("blue") is None


def test_detect_block_unknown_color_returns_none():
    frame = _bgr_frame_with_blob(5, 200, 200, _CX, _CY, _R_BIG)
    with patch("fp.vision.capture_frame", return_value=frame):
        assert vision.detect_block("purple") is None


def test_build_mask_green_blob_hits():
    hsv = _hsv_frame_with_blob(55, 200, 200, _CX, _CY, _R_BIG)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["green"])
    assert mask[_CY, _CX] > 0


def test_detect_all_colors_returns_all_color_keys():
    frame = np.zeros((_TEST_H, _TEST_W, 3), dtype=np.uint8)
    with patch("fp.vision.capture_frame", return_value=frame):
        result = vision.detect_all_colors()
    assert set(result.keys()) == {"red", "blue", "green"}


def test_detect_all_colors_all_none_for_empty_frame():
    frame = np.zeros((_TEST_H, _TEST_W, 3), dtype=np.uint8)
    with patch("fp.vision.capture_frame", return_value=frame):
        result = vision.detect_all_colors()
    assert all(v is None for v in result.values())


def test_detect_all_colors_returns_detection_for_red_blob():
    frame = _bgr_frame_with_blob(5, 200, 200, _CX, _CY, _R_BIG)
    with patch("fp.vision.capture_frame", return_value=frame):
        result = vision.detect_all_colors()
    assert result["red"] is not None
    assert result["red"].label == "red"
    assert result["blue"] is None
    assert result["green"] is None


def test_best_contour_cy_above_max_valid_returns_none():
    # Blob whose centroid lands below VISION_MAX_VALID_CY (600) in the
    # rotated frame is floor glare; _best_contour must reject it.
    cy_low = vision.VISION_MAX_VALID_CY + 50   # 650, comfortably past threshold
    hsv = _hsv_frame_with_blob(5, 200, 200, _CX, cy_low, _R_BIG)
    mask = vision._build_mask(hsv, vision.COLOR_MASKS["red"])
    assert vision._best_contour(mask, "red") is None
