"""
qr_detection.py - QR code detection and arena-pose decoding.

Decodes QR codes placed at known arena locations to provide absolute pose
fixes for the EKF.

QR Payload Format
-----------------
Each QR code encodes a short string: "x=NNN,y=NNN"
  NNN : integer centimetres from the SW corner of the arena.
Example: "x=107,y=107"  (centre of a 213 cm arena)

The QR code's physical position must be pre-measured and stored in
robot.QR_LANDMARKS.  Payload coordinates are the primary source;
QR_LANDMARKS is used as a cross-check and for heading estimation.

Robot pose from QR
------------------
When the camera is calibrated (calibration/camera_matrix.npz exists),
a PnP solve on the four QR corners gives the camera's position and
orientation relative to the QR marker.  Combined with the QR's known
arena position, this yields the robot's arena-frame position and heading.

Without calibration, detect_qr() returns None - the QR marker's position
cannot be safely used as a robot position substitute.

Public API
----------
detect_qr(frame) -> (robot_x_cm, robot_y_cm, heading_deg) | None

Returns the robot's arena-frame pose if a valid QR is found and the PnP
solve succeeds, or None otherwise.
"""

import math
import logging
import re
import numpy as np
import cv2
from typing import Optional, Tuple

from fp import arena
from fp import vision

# QR code physical size in cm (the printed image area, not the whole sticker).
QR_SIZE_CM = 10.0   # update to match the printed QR code's outer dimension


_qr_detector = cv2.QRCodeDetector()


logger = logging.getLogger(__name__)


def _parse_payload(data: str) -> Optional[Tuple[float, float]]:
    """
    Parse 'x=NNN,y=NNN'  ->  (x_cm, y_cm)  or None on failure.
    Accepts integer or float values.
    """
    m = re.fullmatch(r"\s*x\s*=\s*(-?[\d.]+)\s*,\s*y\s*=\s*(-?[\d.]+)\s*",
                     data.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _robot_pose_from_qr(
    corners: np.ndarray,
    qr_x_cm: float,
    qr_y_cm: float,
    qr_size_cm: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute the robot's arena pose from a detected QR code via PnP.

    corners         : (4, 2) float32 - QR pixel corners (TL, TR, BR, BL)
    qr_x_cm, qr_y_cm : known arena position of the QR marker centre (cm)

    PnP gives tvec = position of QR centre in camera frame [tx, ty, tz].
    Camera frame: X right, Y down, Z forward (depth).
    - tz: depth from camera to QR along optical axis
    - tx: lateral offset (positive -> QR is to the right of camera axis)

    In robot frame (X right, Y forward), the QR is at approximately (tx, tz).
    In arena frame with heading θ (0 = north, CW):
      forward dir : (sin θ,  cos θ)
      right   dir : (cos θ, −sin θ)

    robot_x = qr_x − tz·sin θ − tx·cos θ
    robot_y = qr_y − tz·cos θ + tx·sin θ

    Returns (robot_x_cm, robot_y_cm, heading_deg) or None if PnP fails or
    calibration is unavailable.
    """
    if not vision.is_calibrated():
        return None

    half = qr_size_cm / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    img_pts = corners.reshape(4, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts,
        vision.get_camera_matrix(), vision.get_dist_coeffs(),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return None

    tx = float(tvec[0])
    tz = float(tvec[2])

    # Heading: yaw around camera Y axis (pan angle) derived from the rotation matrix.
    # Standard ZYX Euler yaw extraction: atan2(-R[2,0], R[0,0]).
    # For a pure Y-axis rotation R_y(θ): R[0,0]=cos(θ), R[2,0]=-sin(θ),
    # so atan2(sin(θ), cos(θ)) = θ.  This is more robust than the
    # surface-normal projection (atan2(R[0,2], R[2,2])) under compound rotations.
    R, _ = cv2.Rodrigues(rvec)
    yaw_rad = math.atan2(-R[2, 0], R[0, 0])
    heading_deg = math.degrees(yaw_rad) % 360.0

    theta_rad = math.radians(heading_deg)
    robot_x = qr_x_cm - tz * math.sin(theta_rad) - tx * math.cos(theta_rad)
    robot_y = qr_y_cm - tz * math.cos(theta_rad) + tx * math.sin(theta_rad)

    return (robot_x, robot_y, heading_deg)


def detect_qr(frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Attempt to detect and decode a QR code in *frame*.

    Returns (robot_x_cm, robot_y_cm, heading_deg) - the *robot's* arena-frame
    pose, computed from the QR marker's known position and the camera's
    relative pose via PnP.

    Returns None if:
    - No QR code is detected or the payload is malformed
    - No corner data is available from the detector
    - Camera calibration is absent (PnP unavailable)
    - The PnP solve fails
    """
    data, corners, _ = _qr_detector.detectAndDecode(frame)

    if not data:
        return None

    parsed = _parse_payload(data)
    if parsed is None:
        logger.info(f"[QR] Detected but could not parse payload: '{data}'")
        return None

    qr_x, qr_y = parsed

    # Cross-check against known landmarks
    key = f"x={round(qr_x)},y={round(qr_y)}"
    if key not in arena.QR_LANDMARKS:
        logger.info(f"[QR] Unknown landmark key '{key}' - treating decoded coordinates as ground truth")
    else:
        qr_x, qr_y = arena.QR_LANDMARKS[key]

    if corners is None:
        logger.info(f"[QR] Decoded ({qr_x:.1f}, {qr_y:.1f}) but no corner data - skipping")
        return None

    pose = _robot_pose_from_qr(corners.reshape(4, 2), qr_x, qr_y, QR_SIZE_CM)
    if pose is None:
        logger.info(f"[QR] Decoded ({qr_x:.1f}, {qr_y:.1f}) but PnP failed - skipping")
        return None

    robot_x, robot_y, heading_deg = pose
    logger.info(f"[QR] Robot pose: ({robot_x:.1f}, {robot_y:.1f}) cm  {heading_deg:.1f} deg  "
          f"[landmark ({qr_x:.1f}, {qr_y:.1f})]")
    return (robot_x, robot_y, heading_deg)
