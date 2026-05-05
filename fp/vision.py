# Camera capture and HSV blob detection.
# COLOR_MASKS thresholds were tuned under arena lighting - do not retune in simulation.
import os
import time
from dataclasses import dataclass
from typing import Dict
import cv2
import numpy as np

from picamera2 import Picamera2
from libcamera import controls

# Frame and FOV
SENSOR_RESOLUTION   = (1280, 720)
CAMERA_HFOV_DEG     = 41.2

# HSV ranges per colour, OR'd. Two ranges for red (hue wraps).
COLOR_MASKS = {
    "red": [
                (165, 100, 80, 179, 235, 255),
                (0, 100, 80, 5, 235, 255)
    ],
    "blue": [
                (95, 120, 100, 120, 255, 255)
    ],
    "green": [
                (42, 110, 80, 82, 255, 255)
    ],
}

# Detection thresholds
MIN_BLOB_AREA       = {"red": 10_000, "blue": 10_000, "green": 10_000}
IN_REACH_TOLERANCE  = {"red": 150_000, "blue": 150_000, "green": 150_000}
TOO_CLOSE_TOLERANCE = {"red": 20_000, "blue": 20_000, "green": 20_000}

_TIMELAPSE_DIR = "time_lapse"
_last_timelapse_t: float = 0.0

# Camera intrinsic calibration
_CAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "calibration", "camera_matrix.npz",
)
_camera_matrix: np.ndarray  = None
_dist_coeffs: np.ndarray    = None
_calibrated: bool           = False

try:
    _cal = np.load(_CAL_PATH)
    _camera_matrix = _cal["camera_matrix"]
    _dist_coeffs = _cal["dist_coeffs"]
    _calibrated = True
    print(f"[CAM] Loaded camera calibration from {_CAL_PATH}")
except FileNotFoundError:
    print(f"[CAM] Calibration file not found: {_CAL_PATH}")
except (KeyError, Exception) as _cal_err:
    print(f"[CAM] Failed to load camera calibration: {_cal_err!r}")

def is_calibrated() -> bool:
    return _calibrated

def get_camera_matrix() -> np.ndarray:
    return _camera_matrix

@dataclass
class DetectionResult:
    label: str
    cx: int
    cy: int
    area: float

    @property
    def within_reach(self) -> bool:
        return self.area >= IN_REACH_TOLERANCE.get(self.label, float('inf'))

    @property
    def block_too_close(self) -> bool:
        return self.area >= TOO_CLOSE_TOLERANCE.get(self.label, float('inf'))

    @property
    def centroid_error(self) -> int:
        return self.cx - SENSOR_RESOLUTION[0] // 2

_picam2 = None
def init_camera() -> Picamera2:
    global _picam2
    _picam2 = Picamera2()

    # camera is mounted rotated 90 degrees
    cfg = _picam2.create_video_configuration(main={"size": (SENSOR_RESOLUTION[1], SENSOR_RESOLUTION[0]), "format": "RGB888"})
    _picam2.configure(cfg)
    _picam2.start()
    _picam2.set_controls({"AwbMode": controls.AwbModeEnum.Auto})

    time.sleep(3)  # AE/AWB settle
    return _picam2


def release_camera() -> None:
    global _picam2
    if _picam2 is not None:
        _picam2.stop()

def capture_frame() -> np.ndarray:
    global _last_timelapse_t

    # camera is mounted rotated 90 degrees
    bgr = _picam2.capture_array()
    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    now = time.monotonic()
    if now - _last_timelapse_t > 1.0:
        _last_timelapse_t = now
        os.makedirs(_TIMELAPSE_DIR, exist_ok=True)
        fname = os.path.join(_TIMELAPSE_DIR, f"{now:.3f}.jpg")
        cv2.imwrite(fname, bgr)

    return bgr


# Detection

def _build_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) in ranges:
        lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

    return combined


def _best_contour(mask: np.ndarray, label: str) -> tuple[np.ndarray, float, int, int]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA.get(label, float('inf')):
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        candidates.append((c, area, cx, cy))

    if not candidates:
        return None
    
    return max(candidates, key=lambda t: t[1])


def detect_all_colors() -> Dict[str, DetectionResult]:
    frame = capture_frame()
    if frame is None:
        print("[VISION] detect_all_colors received None frame for detection")
        return {c: None for c in COLOR_MASKS}
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out: Dict[str, DetectionResult] = {}
    for label, ranges in COLOR_MASKS.items():
        mask = _build_mask(hsv, ranges)
        result = _best_contour(mask, label)

        if result is None:
            out[label] = None
        else:
            _, area, cx, cy = result
            out[label] = DetectionResult(label=label, cx=cx, cy=cy, area=area)
    return out


def detect_block(target: str) -> DetectionResult:
    if target not in COLOR_MASKS:
        print(f"[VISION] Invalid target color '{target}' for detection")
        return None

    dets = detect_all_colors()
    return dets.get(target)
