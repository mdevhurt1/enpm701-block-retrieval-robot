import math
import sys
import os

# Add parent directory to path so we can import final_project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # applies hw_stub before ekf imports robot

import numpy as np
from fp import ekf as ekf_module
from fp.ekf import EKF


# Convenience: radians <-> degrees
def r(d): return math.radians(d)
def d(r_): return math.degrees(r_)


class TestPredict:

    def test_pure_forward_1m(self):
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=0.0,
                  P0=np.diag([1.0, 1.0, (r(1))**2]))
        ekf.predict(d_cm=100.0, omega_rad_s=0.0, dt=1.0)
        x, y, hdg = ekf.get_pose()
        assert abs(x) < 0.01,       f"x should be ~0, got {x:.4f}"
        assert abs(y - 100.0) < 0.01, f"y should be ~100, got {y:.4f}"
        assert abs(hdg) < 0.1,      f"heading should be ~0°, got {hdg:.4f}°"

    def test_pure_forward_east(self):
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=90.0)
        ekf.predict(d_cm=100.0, omega_rad_s=0.0, dt=1.0)
        x, y, hdg = ekf.get_pose()
        assert abs(x - 100.0) < 0.1, f"x should be ~100, got {x:.4f}"
        assert abs(y) < 0.1,         f"y should be ~0, got {y:.4f}"

    def test_covariance_grows_linearly(self):
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=0.0,
                  P0=np.zeros((3, 3)))
        dt = 0.1
        for _ in range(10):
            ekf.predict(d_cm=0.0, omega_rad_s=0.0, dt=dt)
        P = ekf.get_covariance()
        # After 10 * dt = 1 s, P should equal Q * 1.0
        Q = ekf_module._Q_DEFAULT
        np.testing.assert_allclose(P, Q * 1.0, rtol=1e-9)

    def test_reverse_move(self):
        ekf = EKF(x0=0.0, y0=50.0, heading_deg=0.0)
        ekf.predict(d_cm=-20.0, omega_rad_s=0.0, dt=0.5)
        _, y, _ = ekf.get_pose()
        assert abs(y - 30.0) < 0.1, f"y should be ~30, got {y:.4f}"


class TestUpdateHeading:

    def test_corrects_heading_drift(self):
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=10.0,
                  R_imu=np.array([[r(0.1)**2]]))  # very tight noise
        ekf.update_heading(0.0)
        _, _, hdg = ekf.get_pose()
        assert abs(hdg) < 1.0, f"heading should be near 0°, got {hdg:.2f}°"

    def test_update_wraps_360(self):
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=358.0)
        ekf.update_heading(1.0)   # small positive update
        _, _, hdg = ekf.get_pose()
        # result should be between 357 and 3 (wraps around 0/360)
        assert hdg > 355 or hdg < 5, f"unexpected heading {hdg:.2f}°"

    def test_sanity_gate_rejects_180_jump(self):
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=0.0)
        accepted = ekf.update_heading(181.0)
        assert accepted is False, "180° jump should be rejected by sanity gate"
        _, _, hdg = ekf.get_pose()
        assert abs(hdg) < 5.0, f"pose should be unchanged after rejection, hdg={hdg:.2f}°"

    def test_mahalanobis_gate_rejects_heading_outlier(self):
        import math as _math
        # Very tight initial uncertainty + tight R → any large innovation → high mahal
        ekf = EKF(x0=0.0, y0=0.0, heading_deg=45.0,
                  P0=np.diag([1.0, 1.0, _math.radians(0.1) ** 2]),
                  R_imu=np.array([[_math.radians(0.1) ** 2]]))
        # 45° → 90°: innovation = 45° which is huge relative to 0.1° std
        accepted = ekf.update_heading(90.0)
        assert accepted is False, "large mahal heading update should be rejected"


class TestUpdateWall:

    def test_south_wall_corrects_y(self):
        from fp import arena as _robot
        ekf = EKF(x0=50.0, y0=25.0, heading_deg=180.0)
        # With ROBOT_SENSOR_OFFSET_CM=10, measured d=12 gives y = 0 + 12 + 10 = 22
        accepted = ekf.update_wall('south', 12.0)
        assert accepted, "south wall update should be accepted"
        _, y, _ = ekf.get_pose()
        expected_y = _robot.ARENA_SOUTH_WALL_Y_CM + 12.0 + _robot.ROBOT_SENSOR_OFFSET_CM
        assert abs(y - expected_y) < 5.0, f"y={y:.2f}, expected ≈{expected_y:.2f}"

    def test_mahalanobis_gate_rejects_outlier(self):
        ekf = EKF(x0=50.0, y0=50.0, heading_deg=90.0,
                  P0=np.diag([1.0, 1.0, r(1)**2]))   # tight uncertainty
        # y is at 50 cm; claim east wall puts us at y≈300 cm - absurd
        accepted = ekf.update_wall('east', 300.0)
        assert not accepted, "10-sigma outlier should be rejected"


class TestUpdateQR:

    def test_qr_corrects_all_states(self):
        # Large P0 → relaxed gate, tight R_qr → strong correction
        ekf = EKF(x0=48.0, y0=78.0, heading_deg=88.0,
                  P0=np.diag([1.0, 1.0, r(1.0)**2]),
                  R_qr=np.diag([0.01, 0.01, r(0.1)**2]))  # very tight
        accepted = ekf.update_qr(x_cm=50.0, y_cm=80.0, heading_deg=90.0)
        assert accepted
        x, y, hdg = ekf.get_pose()
        assert abs(x - 50.0) < 5.0
        assert abs(y - 80.0) < 5.0
        assert abs(hdg - 90.0) < 5.0
