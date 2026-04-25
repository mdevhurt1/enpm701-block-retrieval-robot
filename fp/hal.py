import RPi.GPIO as gpio
import serial
import threading
import time
import math
import logging
from collections import deque
from typing import Optional

# GPIO pin assignments  (BOARD numbering)

LEFT_FORWARD  = 37
LEFT_REVERSE  = 35
RIGHT_REVERSE = 33
RIGHT_FORWARD = 31

ENC_A = 7
ENC_B = 12

TRIG_FRONT  = 16
ECHO_FRONT  = 18

SERVO_PIN      = 36
SERVO_HZ       = 50
GRIPPER_OPEN   = 7.5
GRIPPER_CLOSED = 3.5

# Encoder odometry constants
COUNTS_PER_REV      = 20
WHEEL_DIAMETER_MM   = 65.0
WHEEL_CIRCUM_CM     = math.pi * WHEEL_DIAMETER_MM / 10.0
COUNTS_PER_CM       = COUNTS_PER_REV / WHEEL_CIRCUM_CM
WHEELBASE_CM        = 15.0

# Motor PWM
PWM_FREQ            = 50
PWM_DUTY_STRAIGHT   = 25
PWM_DUTY_TURN       = 50

# Safety
WALL_CLEARANCE_CM   = 15.0

# IMU serial
IMU_PORT          = '/dev/ttyUSB0'
IMU_BAUD          = 115200
HEADING_TOLERANCE = 1.0

_ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403}

logger = logging.getLogger(__name__)


# Robot class — owns all mutable hardware state

class Robot:
    """Encapsulates all mutable HAL state.  One singleton per process."""

    def __init__(self):
        self._servo_pwm    = None
        self._motor_pwm    = {}
        self._enc_counts   = {}
        self._enc_lock     = threading.Lock()
        self._carrying_block = False

        # Wall-clearance consecutive-None counter
        self._wall_none_count: int = 0
        self._front_readings = deque(maxlen=5)

        # IMU state
        self._ser           = None
        self._current_hdg   = 0.0
        self._last_hdg_time = 0.0
        self._hdg_lock      = threading.Lock()
        self._imu_ready     = threading.Event()
        self._imu_stop      = threading.Event()
        self._imu_thread    = None

    # Encoder callbacks

    def _bump(self, pin: int) -> None:
        with self._enc_lock:
            self._enc_counts[pin] = self._enc_counts.get(pin, 0) + 1

    def read_encoders(self) -> tuple:
        with self._enc_lock:
            return (self._enc_counts.get(ENC_A, 0), self._enc_counts.get(ENC_B, 0))

    # GPIO init / cleanup

    def init(self) -> None:
        gpio.setmode(gpio.BOARD)

        for pin in (LEFT_FORWARD, LEFT_REVERSE, RIGHT_REVERSE, RIGHT_FORWARD):
            gpio.setup(pin, gpio.OUT)
            gpio.output(pin, False)
            self._motor_pwm[pin] = gpio.PWM(pin, PWM_FREQ)
            self._motor_pwm[pin].start(0)

        gpio.setup(ENC_A, gpio.IN, pull_up_down=gpio.PUD_UP)
        gpio.setup(ENC_B, gpio.IN, pull_up_down=gpio.PUD_UP)
        self._enc_counts[ENC_A] = 0
        self._enc_counts[ENC_B] = 0
        gpio.add_event_detect(ENC_A, gpio.RISING, callback=self._bump, bouncetime=10)
        gpio.add_event_detect(ENC_B, gpio.RISING, callback=self._bump, bouncetime=10)

        gpio.setup(TRIG_FRONT, gpio.OUT)
        gpio.setup(ECHO_FRONT, gpio.IN)
        gpio.output(TRIG_FRONT, False)

        gpio.setup(SERVO_PIN, gpio.OUT)
        self._servo_pwm = gpio.PWM(SERVO_PIN, SERVO_HZ)
        self._servo_pwm.start(GRIPPER_OPEN)

    def cleanup(self) -> None:
        self.stop_motors()
        for pwm in self._motor_pwm.values():
            pwm.stop()
        self._motor_pwm.clear()
        if self._servo_pwm is not None:
            self._servo_pwm.stop()
        gpio.cleanup()

    # Motors

    def stop_motors(self) -> None:
        for pwm in self._motor_pwm.values():
            pwm.ChangeDutyCycle(0)

    def set_individual_duties(self, left_pin: int, left_duty: float,
                              right_pin: int, right_duty: float) -> None:
        self._motor_pwm[left_pin].ChangeDutyCycle(left_duty)
        self._motor_pwm[right_pin].ChangeDutyCycle(right_duty)

    def motors_forward(self, duty: float = PWM_DUTY_STRAIGHT) -> None:
        self.stop_motors()
        self._motor_pwm[LEFT_FORWARD].ChangeDutyCycle(duty)
        self._motor_pwm[RIGHT_FORWARD].ChangeDutyCycle(duty)

    def motors_reverse(self, duty: float = PWM_DUTY_STRAIGHT) -> None:
        self.stop_motors()
        self._motor_pwm[LEFT_REVERSE].ChangeDutyCycle(duty)
        self._motor_pwm[RIGHT_REVERSE].ChangeDutyCycle(duty)

    def motors_pivot_cw(self, duty: float = PWM_DUTY_TURN) -> None:
        self.stop_motors()
        self._motor_pwm[LEFT_FORWARD].ChangeDutyCycle(duty)
        self._motor_pwm[RIGHT_REVERSE].ChangeDutyCycle(duty)

    def motors_pivot_ccw(self, duty: float = PWM_DUTY_TURN) -> None:
        self.stop_motors()
        self._motor_pwm[LEFT_REVERSE].ChangeDutyCycle(duty)
        self._motor_pwm[RIGHT_FORWARD].ChangeDutyCycle(duty)

    # Gripper

    def gripper_open(self) -> None:
        self._carrying_block = False
        if self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(GRIPPER_OPEN)
        time.sleep(0.5)

    def gripper_close(self) -> None:
        if self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(GRIPPER_CLOSED)
        time.sleep(0.5)

    def gripper_set(self, duty: float) -> None:
        if self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(duty)

    def set_carrying_block(self, carrying: bool) -> None:
        self._carrying_block = bool(carrying)
        if not carrying:
            self._wall_none_count = 0

    def is_carrying_block(self) -> bool:
        return self._carrying_block

    # Ultrasonic

    @staticmethod
    def _ping(trig_pin: int, echo_pin: int, timeout_s: float = 0.03):
        gpio.output(trig_pin, False)
        time.sleep(0.005)
        gpio.output(trig_pin, True)
        time.sleep(0.00001)
        gpio.output(trig_pin, False)

        t0 = time.monotonic()
        while gpio.input(echo_pin) == 0:
            if time.monotonic() - t0 > timeout_s:
                return None
        pulse_start = time.monotonic()
        while gpio.input(echo_pin) == 1:
            if time.monotonic() - pulse_start > timeout_s:
                return None
        pulse_end = time.monotonic()
        return round((pulse_end - pulse_start) * 17150, 1)

    def distance_front(self):
        reading = self._ping(TRIG_FRONT, ECHO_FRONT)
        self._front_readings.append(reading)
        return reading

    def get_recent_front_readings(self) -> list:
        return list(self._front_readings)

    def _recent_front_readings_str(self) -> str:
        if not self._front_readings:
            return "[]"
        formatted = ["None" if reading is None else f"{reading:.1f}" for reading in self._front_readings]
        return "[" + ", ".join(formatted) + "]"

    def wall_too_close(self) -> bool:
        if self._carrying_block:
            self._wall_none_count = 0
            return False

        d = self.distance_front()
        if d is None:
            self._wall_none_count += 1
            if self._wall_none_count >= 2:
                logger.warning(
                    f"[ULTRA] wall_too_close triggered after {self._wall_none_count} consecutive timeouts; "
                    f"recent front readings={self._recent_front_readings_str()}"
                )
                return True
            return False
        self._wall_none_count = 0
        if d < WALL_CLEARANCE_CM:
            logger.warning(
                f"[ULTRA] wall_too_close triggered by {d:.1f} cm reading "
                f"(< {WALL_CLEARANCE_CM:.1f} cm); recent front readings={self._recent_front_readings_str()}"
            )
            return True
        return False

    # IMU

    def _imu_reader(self) -> None:
        warmup = 0
        while not self._imu_stop.is_set():
            try:
                if self._ser and self._ser.in_waiting > 0:
                    line = self._ser.readline().decode('ascii', errors='ignore').strip()
                    warmup += 1
                    if warmup <= 10:
                        continue
                    try:
                        hdg = float(line)
                        if 0.0 <= hdg < 360.0:
                            with self._hdg_lock:
                                self._current_hdg   = hdg
                                self._last_hdg_time = time.monotonic()
                            self._imu_ready.set()
                    except ValueError:
                        pass
                else:
                    time.sleep(0.005)
            except serial.SerialException as e:
                logger.error(f"[IMU] Serial error: {e}")
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"[IMU] Unexpected error: {e}")

    @staticmethod
    def _find_imu_port(preferred: str = IMU_PORT) -> str:
        try:
            from serial.tools.list_ports import comports
            for info in comports():
                if info.vid in _ARDUINO_VIDS:
                    logger.info(f"[IMU] Auto-discovered port: {info.device} "
                                f"(VID=0x{info.vid:04X} '{info.description}')")
                    return info.device
        except Exception as e:
            logger.warning(f"[IMU] Port auto-discovery failed: {e}")
        logger.info(f"[IMU] Using fallback port: {preferred}")
        return preferred

    def start_imu(self, port: Optional[str] = None, baud: int = IMU_BAUD,
                  timeout_s: int = 10) -> bool:
        if port is None:
            port = self._find_imu_port(IMU_PORT)
        try:
            self._ser = serial.Serial(port, baud, timeout=1)
        except Exception as e:
            logger.error(f"[IMU] Could not open {port}: {e}")
            return False
        self._imu_stop.clear()
        self._imu_thread = threading.Thread(
            target=self._imu_reader, daemon=True, name="imu-reader"
        )
        self._imu_thread.start()
        logger.info("[IMU] Waiting for first reading...")
        if not self._imu_ready.wait(timeout=timeout_s):
            logger.warning("[IMU] No heading received within timeout")
            return False
        time.sleep(0.5)
        logger.info(f"[IMU] Ready - heading: {self.get_heading():.1f} deg")
        return True

    def stop_imu(self) -> None:
        self._imu_stop.set()
        if self._imu_thread:
            self._imu_thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()

    def is_imu_healthy(self, stale_threshold_s: float = 0.5) -> bool:
        with self._hdg_lock:
            if self._last_hdg_time == 0:
                return False
            return (time.monotonic() - self._last_hdg_time) <= stale_threshold_s

    def get_heading(self) -> float:
        with self._hdg_lock:
            if self._last_hdg_time == 0.0:
                raise RuntimeError("[IMU] No heading received — is start_imu() complete?")
            age = time.monotonic() - self._last_hdg_time
            if age > 0.5:
                raise RuntimeError(
                    f"[IMU] Heading is {age:.1f}s stale — IMU connection may be lost"
                )
            return self._current_hdg

    def get_heading_or_none(self):
        with self._hdg_lock:
            if self._last_hdg_time == 0.0:
                return None
            if time.monotonic() - self._last_hdg_time > 0.5:
                return None
            return self._current_hdg

    def heading_age_s(self):
        with self._hdg_lock:
            if self._last_hdg_time == 0.0:
                return None
            return time.monotonic() - self._last_hdg_time

# Module-level singleton + shim functions (preserve public API)

_default = Robot()


def read_encoders() -> tuple:
    return _default.read_encoders()


def init() -> None:
    _default.init()


def cleanup() -> None:
    _default.cleanup()


def stop_motors() -> None:
    _default.stop_motors()


def set_individual_duties(left_pin: int, left_duty: float,
                          right_pin: int, right_duty: float) -> None:
    _default.set_individual_duties(left_pin, left_duty, right_pin, right_duty)


def motors_forward(duty: float = PWM_DUTY_STRAIGHT) -> None:
    _default.motors_forward(duty)


def motors_reverse(duty: float = PWM_DUTY_STRAIGHT) -> None:
    _default.motors_reverse(duty)


def motors_pivot_cw(duty: float = PWM_DUTY_TURN) -> None:
    _default.motors_pivot_cw(duty)


def motors_pivot_ccw(duty: float = PWM_DUTY_TURN) -> None:
    _default.motors_pivot_ccw(duty)


def gripper_open() -> None:
    _default.gripper_open()


def gripper_close() -> None:
    _default.gripper_close()


def gripper_set(duty: float) -> None:
    _default.gripper_set(duty)


def set_carrying_block(carrying: bool) -> None:
    _default.set_carrying_block(carrying)


def is_carrying_block() -> bool:
    return _default.is_carrying_block()


def distance_front():
    return _default.distance_front()


def wall_too_close() -> bool:
    return _default.wall_too_close()


def start_imu(port: Optional[str] = None, baud: int = IMU_BAUD, timeout_s: int = 10) -> bool:
    return _default.start_imu(port, baud, timeout_s)


def stop_imu() -> None:
    _default.stop_imu()


def is_imu_healthy(stale_threshold_s: float = 0.5) -> bool:
    return _default.is_imu_healthy(stale_threshold_s)


def get_heading() -> float:
    return _default.get_heading()


def get_heading_or_none():
    return _default.get_heading_or_none()


def heading_age_s():
    return _default.heading_age_s()


