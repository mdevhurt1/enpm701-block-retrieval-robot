"""
ekf.py - Extended Kalman Filter for 2-D robot localization.

State: x = [x_cm, y_cm, theta_rad]
  x_cm, y_cm : robot centre position in arena frame (SW corner = origin)
  theta_rad  : heading in radians, arena convention 0 = north (+Y), positive CW

Motion model (differential-drive, predict step):
  x_{k+1}     = x_k + d_k * sin(theta_k)
  y_{k+1}     = y_k + d_k * cos(theta_k)
  theta_{k+1} = theta_k + omega_k * dt

Observation sources and their H matrices:
  - IMU absolute heading  : H = [0, 0, 1]
  - Wall distance         : H selects x or y row based on wall name
  - Camera PnP block pose : H = I_{2x3} (x and y rows)
  - QR code absolute fix  : H = I_{3x3} (all three rows)

All angles internally in radians; public API accepts/returns degrees.

Public API
----------
EKF(x0, y0, heading_deg, P0, Q, R_imu, R_wall, R_block, R_qr)
predict(d_cm, omega_rad_s, dt)
update_heading(heading_deg, R_override=None)
update_wall(wall, dist_cm, R_override=None)
update_qr(x_cm, y_cm, heading_deg, R_override=None)
get_pose()        -> (x_cm, y_cm, heading_deg)
get_covariance()  -> P  (3x3 numpy array)
max_variance()    -> float  (largest diagonal element of P)
"""

import math
import logging
import threading
from typing import Optional
import numpy as np
from fp import arena


# Default noise parameters - tune on the physical robot.
# Units: cm^2 for position, rad^2 for heading.

# Process noise (motion model uncertainty per second)
_Q_DEFAULT = np.diag([2.0, 2.0, math.radians(2.0) ** 2])   # [cm^2, cm^2, rad^2]

# IMU absolute heading observation noise (BNO055 fusion heading, ~1 deg std)
_R_IMU_DEFAULT = np.array([[math.radians(1.5) ** 2]])

# Ultrasonic wall distance noise (~2 cm std after trimmed mean)
_R_WALL_DEFAULT = np.array([[4.0]])   # cm^2

# Camera PnP block position noise (~5 cm std)
_R_BLOCK_DEFAULT = np.diag([25.0, 25.0])   # cm^2

# QR code full-pose noise (~3 cm std position, ~2 deg std heading)
_R_QR_DEFAULT = np.diag([9.0, 9.0, math.radians(2.0) ** 2])

# Mahalanobis distance gate: reject measurements with chi^2 > threshold.
# chi^2 (1 dof, 95%): 3.84 ; (2 dof, 95%): 5.99 ; (3 dof, 95%): 7.81
_MAHAL_GATE_1DOF = 7.0   # conservative; reduce false rejections near walls
_MAHAL_GATE_2DOF = 10.0
_MAHAL_GATE_3DOF = 12.0




logger = logging.getLogger(__name__)


class EKF:
    """
    3-DoF Extended Kalman Filter for differential-drive robot localization.

    Coordinate convention matches the rest of the codebase:
      arena frame: SW corner = (0, 0), +X east, +Y north, 0 deg heading = north.
    Internally all angles are stored in radians.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading_deg: float = 0.0,
        P0: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R_imu: Optional[np.ndarray] = None,
        R_wall: Optional[np.ndarray] = None,
        R_block: Optional[np.ndarray] = None,
        R_qr: Optional[np.ndarray] = None,
    ):
        theta0 = math.radians(heading_deg)
        self._x = np.array([x0, y0, theta0], dtype=float)
        self._P = np.diag([100.0, 100.0, math.radians(10.0) ** 2]) if P0 is None else P0.copy()
        self._Q = _Q_DEFAULT.copy() if Q is None else Q.copy()
        self._R_imu   = _R_IMU_DEFAULT.copy()   if R_imu   is None else R_imu.copy()
        self._R_wall  = _R_WALL_DEFAULT.copy()  if R_wall  is None else R_wall.copy()
        self._R_block = _R_BLOCK_DEFAULT.copy() if R_block is None else R_block.copy()
        self._R_qr    = _R_QR_DEFAULT.copy()    if R_qr    is None else R_qr.copy()
        self._lock    = threading.Lock()

    # Prediction step

    def predict(self, d_cm: float, omega_rad_s: float, dt: float):
        """
        Propagate the EKF state forward by one time step.

        d_cm        : signed displacement estimated from encoders (+ = forward)
        omega_rad_s : yaw rate from IMU gyroscope (rad/s, + = clockwise)
        dt          : time elapsed since last predict call (seconds)
        """
        if dt <= 0:
            return

        with self._lock:
            x, y, theta = self._x
            Q_eff = self._Q

            # Motion model
            new_theta = theta + omega_rad_s * dt
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            new_x = x + d_cm * sin_t
            new_y = y + d_cm * cos_t

            # Jacobian F = df/dx
            F = np.array([
                [1.0, 0.0, d_cm * cos_t],
                [0.0, 1.0, -d_cm * sin_t],
                [0.0, 0.0,  1.0],
            ])

            # Covariance propagation
            new_P = F @ self._P @ F.T + Q_eff * dt

            self._x = np.array([new_x, new_y, new_theta])
            self._P = new_P

    # Update steps

    def _kf_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray,
                   gate: float) -> bool:
        """
        Generic EKF update.  Returns True if the measurement was accepted.
        """
        with self._lock:
            innovation = z - H @ self._x
            # Wrap heading components of innovation to [-pi, +pi]
            for i, row in enumerate(H):
                if abs(row[2]) > 0.5:   # this row involves theta
                    innovation[i] = (innovation[i] + math.pi) % (2 * math.pi) - math.pi

            S = H @ self._P @ H.T + R

            # Mahalanobis distance gate
            mahal_sq = float(innovation.T @ np.linalg.solve(S, innovation))
            if mahal_sq > gate:
                logger.warning(f"[EKF] Measurement rejected (mahal^2={mahal_sq:.2f} > gate={gate:.2f})")
                return False

            K = self._P @ H.T @ np.linalg.inv(S)
            self._x = self._x + K @ innovation
            self._P = (np.eye(3) - K @ H) @ self._P

            # Keep P symmetric
            self._P = (self._P + self._P.T) / 2.0
            return True

    def update_heading(self, heading_deg: float, R_override: Optional[np.ndarray] = None):
        """
        Absolute heading observation from IMU (BNO055 fusion heading).
        heading_deg : arena-frame heading in degrees (0 = north, CW positive)

        Rejects the measurement if:
          - the wrapped innovation exceeds 170 degrees (sensor sanity check), or
          - the Mahalanobis distance exceeds _MAHAL_GATE_1DOF (statistical gate).
        """
        z = np.array([math.radians(heading_deg)])
        H = np.array([[0.0, 0.0, 1.0]])
        R = R_override if R_override is not None else self._R_imu

        # Fast pre-check: reject obviously implausible heading jumps before
        # computing the Mahalanobis distance (avoids acquiring the lock).
        with self._lock:
            raw_innovation = float(z[0] - self._x[2])
        wrapped = (raw_innovation + math.pi) % (2 * math.pi) - math.pi
        if abs(wrapped) > math.radians(170):
            logger.warning(
                f"[EKF] Heading update rejected: innovation {math.degrees(wrapped):.1f}° "
                f"exceeds 170° sanity limit"
            )
            return False

        return self._kf_update(z, H, R, _MAHAL_GATE_1DOF)

    def update_wall(self, wall: str, dist_cm: float,
                    R_override: Optional[np.ndarray] = None) -> bool:
        """
        Ultrasonic wall-distance observation.

        The robot must be facing the named wall when this is called.
        dist_cm is the raw sensor reading (sensor face to wall).
        The robot centre is ROBOT_SENSOR_OFFSET_CM behind the sensor face.

        Wall geometry (SW corner = origin):
          south -> robot_y = dist_cm + SENSOR_OFFSET
          north -> robot_y = ARENA_H   - dist_cm - SENSOR_OFFSET
          west  -> robot_x = dist_cm + SENSOR_OFFSET
          east  -> robot_x = ARENA_W   - dist_cm - SENSOR_OFFSET
        """
        wall = wall.lower()
        if wall not in ('north', 'south', 'east', 'west'):
            return False

        offset = arena.ROBOT_SENSOR_OFFSET_CM
        W = arena.ARENA_WIDTH_CM
        H = arena.ARENA_HEIGHT_CM

        if wall == 'south':
            z_val = dist_cm + offset           # expected robot_y
            H_row = np.array([0.0, 1.0, 0.0])
        elif wall == 'north':
            z_val = H - dist_cm - offset       # expected robot_y
            H_row = np.array([0.0, 1.0, 0.0])
        elif wall == 'west':
            z_val = dist_cm + offset           # expected robot_x
            H_row = np.array([1.0, 0.0, 0.0])
        else:  # east
            z_val = W - dist_cm - offset       # expected robot_x
            H_row = np.array([1.0, 0.0, 0.0])

        z = np.array([z_val])
        H_mat = H_row.reshape(1, 3)
        R = R_override if R_override is not None else self._R_wall
        return self._kf_update(z, H_mat, R, _MAHAL_GATE_1DOF)

    def update_qr(self, x_cm: float, y_cm: float, heading_deg: float,
                  R_override: Optional[np.ndarray] = None) -> bool:
        """
        QR code absolute pose fix.
        x_cm, y_cm   : decoded arena-frame robot position (cm)
        heading_deg  : decoded arena-frame robot heading (degrees)
        """
        theta = math.radians(heading_deg)
        z = np.array([x_cm, y_cm, theta])
        H = np.eye(3)
        R = R_override if R_override is not None else self._R_qr
        return self._kf_update(z, H, R, _MAHAL_GATE_3DOF)

    # Accessors

    def get_pose(self) -> tuple:
        with self._lock:
            return (
                float(self._x[0]),
                float(self._x[1]),
                math.degrees(float(self._x[2])) % 360.0,
            )

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the 3x3 covariance matrix P."""
        with self._lock:
            return self._P.copy()

    def max_variance(self) -> float:
        """Return the largest diagonal element of P (cm^2 or rad^2)."""
        with self._lock:
            return float(np.max(np.diag(self._P)))

    def set_pose(self, x_cm: float, y_cm: float, heading_deg: float,
                 P: Optional[np.ndarray] = None):
        with self._lock:
            self._x = np.array([x_cm, y_cm, math.radians(heading_deg)], dtype=float)
            if P is not None:
                self._P = P.copy()
