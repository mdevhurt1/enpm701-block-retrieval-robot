import cv2
import math
import logging
import numpy as np
import os
import time
from picamera2 import Picamera2
try:
    from libcamera import Transform as _Transform
    _VFLIP_TRANSFORM = _Transform(vflip=1)
except ImportError:          # non-Pi dev environment
    _VFLIP_TRANSFORM = None
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Camera intrinsic calibration (loaded once at import time if available)

_CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "calibration", "camera_matrix.npz")

_camera_matrix: Optional[np.ndarray] = None
_dist_coeffs:   Optional[np.ndarray] = None
_calibrated:    bool = False

logger = logging.getLogger(__name__)

try:
    _cal = np.load(_CAL_PATH)
    _camera_matrix = _cal["camera_matrix"]
    _dist_coeffs   = _cal["dist_coeffs"]
    _calibrated    = True
    logger.info(f"[CAM] Loaded camera calibration from {_CAL_PATH}")
except FileNotFoundError:
    logger.warning(f"[CAM] Calibration file not found: {_CAL_PATH} - PnP disabled")
except (KeyError, Exception) as _cal_err:
    logger.warning(f"[CAM] Failed to load camera calibration from {_CAL_PATH}: "
                   f"{_cal_err!r} - PnP disabled")



def is_calibrated() -> bool:
    return _calibrated


def get_camera_matrix() -> Optional[np.ndarray]:
    return _camera_matrix


def get_dist_coeffs() -> Optional[np.ndarray]:
    return _dist_coeffs

# Camera config

FRAME_WIDTH  = 960
FRAME_HEIGHT = 720

# HSV colour masks
# Tune these with calibration/hsv_tuner.py under actual arena lighting.
# Each entry is a list of (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) tuples.
# Multiple ranges are OR-ed (needed for red, which wraps the hue circle).

COLOR_MASKS = {
    # Test 2 values
    "red": [
        (161, 76, 121, 179, 232, 217),
        (0, 59, 65, 19, 255, 255)
    ],
    "green": [
        (62, 75, 129, 87, 221, 245)
    ],
    "blue": [
        (76, 104, 93, 121, 255, 235)
    ],
}

'''
{
    # Test 1 values
    "red": [
        (170, 161, 54, 179, 245, 114),
        (0, 161, 54, 19, 255, 255)
    ],
    "green": [
        (46, 111, 51, 103, 225, 255)
    ],
    "blue": [
        (77, 122, 67, 164, 255, 255)
    ],
}
'''

# Minimum contour area to be considered a real block (filters noise).
# Increase if false positives appear; decrease if blocks are missed at distance.
MIN_BLOB_AREA = 20000   # pixels^2

# When blob area exceeds this, the block is within gripper reach.
PICKUP_AREA_THRESHOLD = 36000   # pixels^2

# Horizontal tolerance (pixels) around frame centre for "centred" check in within_reach.
CENTERED_TOLERANCE = FRAME_WIDTH // 24   # +/-80 px for a 1920-wide frame

# Camera helpers

def init_camera() -> Picamera2:
    """Initialise and start the Picamera2. Call once at startup."""
    picam2 = Picamera2()
    _transform_kwarg = {"transform": _VFLIP_TRANSFORM} if _VFLIP_TRANSFORM is not None else {}
    cfg = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        **_transform_kwarg
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(2)   # allow auto-exposure and AWB to settle

    # Lock exposure and white balance so HSV thresholds stay consistent.
    metadata = picam2.capture_metadata()
    picam2.set_controls({
        "AeEnable":     False,
        "AwbEnable":    False,
        "ExposureTime": metadata["ExposureTime"],
        "AnalogueGain": metadata["AnalogueGain"],
        "ColourGains":  metadata["ColourGains"],
    })
    time.sleep(0.5)  # allow controls to take effect
    logger.info(f"[CAM] Exposure locked: {metadata['ExposureTime']} µs  "
          f"gain={metadata['AnalogueGain']:.2f}  "
          f"awb={tuple(round(g, 2) for g in metadata['ColourGains'])}")
    return picam2

def release_camera(picam2: Picamera2):
    picam2.stop()

def capture_frame(picam2: Picamera2) -> np.ndarray:
    """Return a BGR frame (OpenCV convention). Picamera2 stores RGB888 as BGR in memory.

    The vertical flip for the upside-down camera mount is applied in the Picamera2
    Transform config (zero CPU cost). A software fallback runs if libcamera is
    unavailable (dev machine without the libcamera Python binding).
    """
    bgr = picam2.capture_array()
    if _VFLIP_TRANSFORM is None:
        bgr = cv2.flip(bgr, 0)  # software fallback: libcamera Transform not available
    return bgr

# Detection

@dataclass
class DetectionResult:
    label: str
    cx: int
    cy: int
    area: float

    @property
    def within_reach(self) -> bool:
        """True when the block is within gripper range: centred horizontally
        (within CENTERED_TOLERANCE pixels of frame centre) and blob area
        at or above PICKUP_AREA_THRESHOLD (area is a distance proxy —
        larger area means the block is closer)."""
        centred = abs(self.cx - FRAME_WIDTH // 2) <= CENTERED_TOLERANCE
        area_is_close = self.area >= PICKUP_AREA_THRESHOLD
        return centred and area_is_close

    @property
    def centroid_error(self) -> int:
        """Signed pixel offset from frame centre (+ = block is right of centre)."""
        return self.cx - FRAME_WIDTH // 2


def _build_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) in ranges:
        lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
    return combined


def _best_contour(mask: np.ndarray):
    """Return (contour, area, cx, cy) for the largest contour, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_BLOB_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return c, area, cx, cy


def detect_all_colors(frame: np.ndarray) -> Dict[str, Optional["DetectionResult"]]:
    """Detect all colours from a single BGR frame using one HSV conversion.

    Returns a dict mapping each colour name in COLOR_MASKS to its DetectionResult,
    or None if that colour was not found above MIN_BLOB_AREA.

    Use this instead of calling detect_block() per colour to avoid redundant
    BGR→HSV conversions when multiple colours must be checked in the same frame.
    """
    if frame is None:
        return {c: None for c in COLOR_MASKS}
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results: Dict[str, Optional[DetectionResult]] = {}
    for label, ranges in COLOR_MASKS.items():
        mask = _build_mask(hsv, ranges)
        result = _best_contour(mask)
        if result is None:
            results[label] = None
        else:
            _, area, cx, cy = result
            results[label] = DetectionResult(label=label, cx=cx, cy=cy, area=area)
    return results


def detect_block(frame: np.ndarray,
                 target_color: Optional[str] = None) -> Optional[DetectionResult]:
    """Return the best-matching block detection, or None.

    Delegates to detect_all_colors() as the single detection primitive to
    avoid duplicating the BGR→HSV conversion or the contour logic.

    target_color: if given, only that colour is considered.  If the name is
    not in COLOR_MASKS a warning is logged and all colours are scanned.
    """
    if target_color and target_color not in COLOR_MASKS:
        logger.info(
            "[VISION] detect_block: unknown color '%s' - scanning all colors",
            target_color,
        )
        target_color = None
    dets = detect_all_colors(frame)
    if target_color:
        return dets.get(target_color)
    return max((d for d in dets.values() if d), key=lambda d: d.area, default=None)


def annotate_frame(frame: np.ndarray, detection: Optional[DetectionResult]) -> np.ndarray:
    out = frame.copy()
    if detection is not None:
        color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}.get(
            detection.label, (255, 255, 255)
        )
        cv2.circle(out, (detection.cx, detection.cy), 8, color_bgr, -1)
        cv2.putText(
            out,
            f"{detection.label}  area={int(detection.area)}",
            (detection.cx + 10, detection.cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2,
        )
        # Frame centre reference line
        cx_frame = FRAME_WIDTH // 2
        cv2.line(out, (cx_frame, 0), (cx_frame, FRAME_HEIGHT), (200, 200, 200), 1)
    return out


# PnP-based block pose estimation (requires calibrated camera)

# Nominal block dimensions (cm).  All faces are assumed square.
BLOCK_SIZE_CM = 6.0

def estimate_block_pose(
    frame: np.ndarray,
    detection: "DetectionResult",
    robot_pose: Tuple[float, float, float],
    block_size_cm: float = BLOCK_SIZE_CM,
) -> Optional[Tuple[float, float]]:
    """
    Estimate the arena-frame position of a detected block using a bearing +
    pixel-height range estimate.

    This replaces the previous axis-aligned bounding-rect PnP approach, which
    was geometrically incorrect because the bounding rect is biased by block yaw.
    Instead:
      1. cv2.minAreaRect gives a rotated bounding box; its shorter side is the
         least-biased estimate of the block's visible height at any yaw angle.
      2. Range from pixel height:
           range_cm ≈ block_size_cm * fy / pixel_height
      3. Bearing from centroid x-offset relative to principal point:
           bearing_rad = atan2(cx - cx_principal, fx)
      4. Reject if range > 150 cm (exceeds arena max) or aspect ratio > 1.5
         (block is occluded / tipped / not a cube face).

    Returns (x_arena_cm, y_arena_cm), or None if calibration is unavailable,
    the detection is rejected, or the estimate is out of range.

    robot_pose : (x_cm, y_cm, heading_deg) from navigation.get_pose() or EKF.
    """
    if not _calibrated or _camera_matrix is None:
        return None

    # Extract focal lengths and principal point from calibration matrix
    fx = float(_camera_matrix[0, 0])
    fy = float(_camera_matrix[1, 1])
    cx_principal = float(_camera_matrix[0, 2])

    # Re-detect contour for precise geometry
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = COLOR_MASKS.get(detection.label)
    if ranges is None:
        return None
    mask = _build_mask(hsv, ranges)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_BLOB_AREA:
        return None

    # Rotated bounding box - use the shorter side as pixel_height (least biased to yaw)
    (_, _), (w_rot, h_rot), _ = cv2.minAreaRect(c)
    if w_rot < 1.0 or h_rot < 1.0:
        return None

    pixel_height = min(w_rot, h_rot)
    pixel_width  = max(w_rot, h_rot)

    # Reject occluded/tipped blocks: aspect ratio > 1.5
    aspect = pixel_width / pixel_height
    if aspect > 1.5:
        return None

    # 1. Range estimate from pixel height
    range_cm = block_size_cm * fy / pixel_height

    # 2. Reject implausible ranges
    if range_cm > 150.0 or range_cm < 1.0:
        return None

    # 3. Bearing estimate from centroid x-offset
    bearing_rad = math.atan2(detection.cx - cx_principal, fx)

    # 4. Transform to arena frame
    robot_x, robot_y, robot_hdg_deg = robot_pose
    hdg_rad = math.radians(robot_hdg_deg)

    # Block is at bearing_rad left/right of robot forward direction.
    # Arena: forward = (sin(hdg), cos(hdg)), right = (cos(hdg), -sin(hdg))
    arena_x = robot_x + range_cm * math.sin(hdg_rad + bearing_rad)
    arena_y = robot_y + range_cm * math.cos(hdg_rad + bearing_rad)

    return (arena_x, arena_y)

