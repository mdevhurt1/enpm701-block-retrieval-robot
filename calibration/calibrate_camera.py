"""
calibrate_camera.py - Offline camera intrinsic calibration tool.

Uses a printed checkerboard target to compute the camera matrix K (3x3)
and lens distortion coefficients, then saves them to camera_matrix.npz.

Usage (on the Raspberry Pi, with checkerboard visible to camera):
    python calibration/calibrate_camera.py

Controls:
    SPACE  - capture current frame as a calibration image
    c      - compute calibration from captured images (need >= 15)
    s      - save calibration to camera_matrix.npz and quit
    q      - quit without saving

Output:
    calibration/camera_matrix.npz  (arrays: camera_matrix, dist_coeffs)

Checkerboard:
    Default: 10x7 inner corners (11x8 squares printed on A4/Letter).
    Edit BOARD_W / BOARD_H below if your board differs.
    Square size is only needed for metric 3D reconstruction; for this
    project we only need the intrinsic matrix for undistortion and PnP,
    so SQUARE_SIZE_MM is informational only.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure the project root is on the path so vision.py is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vision

# Checkerboard configuration - edit to match your printed target
BOARD_W        = 10  # inner corners along the width
BOARD_H        = 7   # inner corners along the height
SQUARE_SIZE_MM = 25  # physical square size in mm (informational)

# Minimum accepted calibration images
MIN_IMAGES = 15

# Output path (relative to this script's directory)
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH  = os.path.join(_SCRIPT_DIR, "camera_matrix.npz")

# OpenCV termination criteria for corner sub-pixel refinement
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _make_object_points():
    """Return the 3D coordinates of inner corners on a flat board (Z=0)."""
    objp = np.zeros((BOARD_H * BOARD_W, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
    return objp


def run():
    print("=" * 60)
    print("Camera Intrinsic Calibration")
    print(f"  Board: {BOARD_W}x{BOARD_H} inner corners  |  square: {SQUARE_SIZE_MM} mm")
    print(f"  Target images needed: {MIN_IMAGES}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 60)
    print()

    picam2 = vision.init_camera()
    print("Camera ready.  Controls: SPACE=capture  c=compute  s=save&quit  q=quit\n")

    obj_pts   = []   # 3D points in world space
    img_pts   = []   # 2D points in image space
    objp_tmpl = _make_object_points()
    img_size  = (vision.FRAME_WIDTH, vision.FRAME_HEIGHT)

    camera_matrix  = None
    dist_coeffs    = None
    rpe            = None   # mean reprojection error

    try:
        while True:
            frame = vision.capture_frame(picam2)
            if frame is None:
                print("[CAL] Frame capture failed")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Attempt detection
            found, corners = cv2.findChessboardCorners(
                gray, (BOARD_W, BOARD_H),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            # Draw overlay
            preview = frame.copy()
            cv2.drawChessboardCorners(preview, (BOARD_W, BOARD_H), corners, found)
            status_clr = (0, 255, 0) if found else (0, 0, 255)
            status_txt = f"DETECTED  imgs={len(obj_pts)}" if found else f"NOT FOUND  imgs={len(obj_pts)}"
            cv2.putText(preview, status_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_clr, 2)
            if camera_matrix is not None:
                cv2.putText(preview, f"RPE={rpe:.3f}px", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.imshow("Calibration", preview)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("[CAL] Quit without saving")
                break

            elif key == ord(' '):
                if found:
                    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CRITERIA)
                    obj_pts.append(objp_tmpl)
                    img_pts.append(refined)
                    print(f"[CAL] Captured image {len(obj_pts)}")
                else:
                    print("[CAL] Board not detected - move the target")

            elif key == ord('c'):
                if len(obj_pts) < MIN_IMAGES:
                    print(f"[CAL] Need at least {MIN_IMAGES} images (have {len(obj_pts)})")
                    continue
                print(f"[CAL] Computing calibration from {len(obj_pts)} images...")
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_pts, img_pts, img_size, None, None
                )
                # Mean reprojection error
                total_err = 0.0
                for i, (op, ip) in enumerate(zip(obj_pts, img_pts)):
                    projected, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], mtx, dist)
                    err = cv2.norm(ip, projected, cv2.NORM_L2) / len(projected)
                    total_err += err
                rpe = total_err / len(obj_pts)
                camera_matrix = mtx
                dist_coeffs   = dist
                print(f"[CAL] Calibration complete - mean RPE: {rpe:.4f} px")
                print(f"[CAL] Camera matrix:\n{mtx}")
                print(f"[CAL] Distortion coefficients: {dist.ravel()}")

            elif key == ord('s'):
                if camera_matrix is None:
                    print("[CAL] No calibration computed yet - press 'c' first")
                    continue
                np.savez(OUTPUT_PATH,
                         camera_matrix=camera_matrix,
                         dist_coeffs=dist_coeffs,
                         rpe=np.float64(rpe))
                print(f"[CAL] Saved to {OUTPUT_PATH}  (RPE={rpe:.4f} px)")
                break

    finally:
        cv2.destroyAllWindows()
        vision.release_camera(picam2)
        print("[CAL] Done")


if __name__ == "__main__":
    run()
