import pigpio
import serial
import threading
import time
import math
from collections import deque
from typing import Optional

# Motor pin assignments (BCM numbering, L298N H-Bridge)
_LEFT_FORWARD   = 26  # IN4  (Board 37)
_LEFT_REVERSE   = 19  # IN3  (Board 35)
_RIGHT_REVERSE  = 13  # IN2  (Board 33)
_RIGHT_FORWARD  = 6   # IN1  (Board 31)

# Motor PWM
_PWM_FREQUENCY      = 50
PWM_DUTY_STRAIGHT  = 30
PWM_DUTY_TURN      = 50

# Wheel encoders (BCM, pull-up digital inputs)
_FRONT_LEFT_ENCODER_PIN = 4   # Board 7
_BACK_RIGHT_ENCODER_PIN = 18  # Board 12
_ENCODER_GLITCH_US      = 300

# Ultrasonic sensor
_TRIG               = 23  # Board 16
_ECHO               = 24  # Board 18
_ULTRASONIC_TIMEOUT = 0.02

# Gripper servo (BCM)
_SERVO_PIN         = 16    # Board 36
_GRIPPER_OPEN_US   = 1500  # 7.5% duty at 50 Hz
_GRIPPER_CLOSED_US = 700   # 3.5% duty at 50 Hz

# Encoder odometry constants
_COUNTS_PER_REVOLUTION  = 10
_WHEEL_DIAMETER_MM      = 64.0
_WHEEL_CIRCUMFERENCE_CM = math.pi * _WHEEL_DIAMETER_MM / 10.0
COUNTS_PER_CM           = _COUNTS_PER_REVOLUTION / _WHEEL_CIRCUMFERENCE_CM
WHEELBASE_CM            = 17.0

# IMU
_IMU_PORT               = '/dev/ttyUSB0'
_IMU_BAUD               = 115200
_ARDUINO_VIDS           = {0x2341, 0x2A03, 0x1A86, 0x0403}

class Robot:
    """Encapsulates all mutable HAL state. One singleton per process."""

    def __init__(self):
        self._pi = None
        self._encoder_callbacks = []

        self._front_left_encoder_count  = 0
        self._front_left_encoder_lock   = threading.Lock()

        self._back_right_encoder_count  = 0
        self._back_right_encoder_lock   = threading.Lock()

        self._ultrasonic_readings   = deque(maxlen=5)
        self._last_ultrasonic_time  = 0.0
        self._ultrasonic_lock       = threading.Lock()
        self._ultrasonic_ready      = threading.Event()
        self._ultrasonic_stop       = threading.Event()
        self._ultrasonic_thread     = None

        self._serial                = None
        self._imu_initial_heading   = None
        self._current_heading       = 0.0
        self._last_heading_time     = 0.0
        self._heading_lock          = threading.Lock()
        self._imu_ready             = threading.Event()
        self._imu_stop              = threading.Event()
        self._imu_thread            = None
        self._sample_n              = 0

    def _on_encoder(self, gpio_pin: int, level: int, tick: int) -> None:
        if gpio_pin == _FRONT_LEFT_ENCODER_PIN:
            with self._front_left_encoder_lock:
                self._front_left_encoder_count += 1
        elif gpio_pin == _BACK_RIGHT_ENCODER_PIN:
            with self._back_right_encoder_lock:
                self._back_right_encoder_count += 1
        else:
            print(f"encoder callback called with unknown pin {gpio_pin}")

    def read_back_right_encoder(self) -> int:
        with self._back_right_encoder_lock:
            return self._back_right_encoder_count

    def read_front_left_encoder(self) -> int:
        with self._front_left_encoder_lock:
            return self._front_left_encoder_count

    def init(self) -> None:
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError("[HAL] Could not connect to pigpiod - is the daemon running? (sudo pigpiod)")

        for pin in (_LEFT_FORWARD, _LEFT_REVERSE, _RIGHT_REVERSE, _RIGHT_FORWARD):
            self._pi.set_mode(pin, pigpio.OUTPUT)
            self._pi.write(pin, 0)
            self._pi.set_PWM_frequency(pin, _PWM_FREQUENCY)
            self._pi.set_PWM_range(pin, 100)
            self._pi.set_PWM_dutycycle(pin, 0)

        for pin in (_FRONT_LEFT_ENCODER_PIN, _BACK_RIGHT_ENCODER_PIN):
            self._pi.set_mode(pin, pigpio.INPUT)
            self._pi.set_pull_up_down(pin, pigpio.PUD_UP)
            self._pi.set_glitch_filter(pin, _ENCODER_GLITCH_US)

        self._front_left_encoder_count = 0
        self._back_right_encoder_count = 0

        fl_cb = self._pi.callback(_FRONT_LEFT_ENCODER_PIN, pigpio.RISING_EDGE, self._on_encoder)
        br_cb = self._pi.callback(_BACK_RIGHT_ENCODER_PIN, pigpio.RISING_EDGE, self._on_encoder)
        self._encoder_callbacks = [fl_cb, br_cb]

        self._pi.set_mode(_TRIG, pigpio.OUTPUT)
        self._pi.set_mode(_ECHO, pigpio.INPUT)
        self._pi.write(_TRIG, 0)

        self._pi.set_mode(_SERVO_PIN, pigpio.OUTPUT)
        self._pi.set_servo_pulsewidth(_SERVO_PIN, _GRIPPER_OPEN_US)

    def cleanup(self) -> None:
        self.stop_motors()

        for cb in self._encoder_callbacks:
            cb.cancel()
        self._encoder_callbacks.clear()

        self._pi.set_servo_pulsewidth(_SERVO_PIN, 0)
        self._pi.stop()

    def stop_motors(self) -> None:
        for pin in (_LEFT_FORWARD, _LEFT_REVERSE, _RIGHT_REVERSE, _RIGHT_FORWARD):
            self._pi.set_PWM_dutycycle(pin, 0)

    def set_individual_duties(self, left_direction: int, left_duty: float,
                              right_direction: int, right_duty: float) -> None:
        if left_direction == +1:
            self._pi.set_PWM_dutycycle(_LEFT_FORWARD, left_duty)
        elif left_direction == -1:
            self._pi.set_PWM_dutycycle(_LEFT_REVERSE, left_duty)
        else:
            print(f"[HAL] Unkown direction {left_direction} requested for the left wheels with duty {left_duty}")

        if right_direction == +1:
            self._pi.set_PWM_dutycycle(_RIGHT_FORWARD, right_duty)
        elif right_direction == -1:
            self._pi.set_PWM_dutycycle(_RIGHT_REVERSE, right_duty)
        else:
            print(f"[HAL] Unkown direction {right_direction} requested for the right wheels with duty {right_duty}")

    def motors_forward(self, duty: float = PWM_DUTY_STRAIGHT) -> None:
        self.stop_motors()
        self._pi.set_PWM_dutycycle(_LEFT_FORWARD, duty)
        self._pi.set_PWM_dutycycle(_RIGHT_FORWARD, duty)

    def motors_reverse(self, duty: float = PWM_DUTY_STRAIGHT) -> None:
        self.stop_motors()
        self._pi.set_PWM_dutycycle(_LEFT_REVERSE, duty)
        self._pi.set_PWM_dutycycle(_RIGHT_REVERSE, duty)

    def motors_pivot_cw(self, duty: float = PWM_DUTY_TURN) -> None:
        self.stop_motors()
        self._pi.set_PWM_dutycycle(_LEFT_FORWARD, duty)
        self._pi.set_PWM_dutycycle(_RIGHT_REVERSE, duty)

    def motors_pivot_ccw(self, duty: float = PWM_DUTY_TURN) -> None:
        self.stop_motors()
        self._pi.set_PWM_dutycycle(_LEFT_REVERSE, duty)
        self._pi.set_PWM_dutycycle(_RIGHT_FORWARD, duty)

    def gripper_open(self) -> None:
        if self._pi:
            self._pi.set_servo_pulsewidth(_SERVO_PIN, _GRIPPER_OPEN_US)

    def gripper_close(self) -> None:
        if self._pi:
            self._pi.set_servo_pulsewidth(_SERVO_PIN, _GRIPPER_CLOSED_US)

    def gripper_set(self, duty: float) -> None:
        if self._pi:
            pulsewidth_us = int(duty / 100.0 * 20000)
            self._pi.set_servo_pulsewidth(_SERVO_PIN, pulsewidth_us)

    def start_ultrasonic(self, timeout_s: int = 10) -> bool:
        self._ultrasonic_thread = threading.Thread(
            target=self._ping, daemon=True, name="ultrasonic-reader"
        )
        self._ultrasonic_thread.start()
        print("[HAL] Waiting for first ultrasonic reading...")
        if not self._ultrasonic_ready.wait(timeout=timeout_s):
            print("[HAL] No ultrasonic reading received within timeout")
            return False
        time.sleep(0.5)
        print(f"[HAL] Ultrasonic ready - distance: {self.get_ultrasonic_distance():.1f} cm")
        return True

    def stop_ultrasonic(self) -> None:
        self._ultrasonic_stop.set()
        if self._ultrasonic_thread:
            self._ultrasonic_thread.join(timeout=2.0)

    def _ping(self) -> None:
        while not self._ultrasonic_stop.is_set():
            self._pi.write(_TRIG, 0)
            time.sleep(0.01)
            self._pi.write(_TRIG, 1)
            time.sleep(0.00001)
            self._pi.write(_TRIG, 0)

            echo_recieved = False
            t0 = time.monotonic()
            while self._pi.read(_ECHO) == 0:
                if time.monotonic() - t0 > _ULTRASONIC_TIMEOUT:
                    break
                else:
                    echo_recieved = True

            pulse_start = time.monotonic()
            while self._pi.read(_ECHO) == 1:
                if time.monotonic() - pulse_start > _ULTRASONIC_TIMEOUT:
                    break

            pulse_end = time.monotonic()
            
            if not echo_recieved:
                continue

            distance_cm = round((pulse_end - pulse_start) * 17150, 2)
            with self._ultrasonic_lock:
                if len(self._ultrasonic_readings) >= 5:
                    self._ultrasonic_readings.popleft()
                    self._ultrasonic_readings.append(distance_cm)
                else:
                    self._ultrasonic_readings.append(distance_cm)

                self._last_ultrasonic_time = time.monotonic()
                self._ultrasonic_ready.set()

    def get_ultrasonic_distance(self) -> float:
        return self._ultrasonic_readings[-1]

    def get_recent_ultrasonic_readings(self) -> list:
        ret = None
        with self._ultrasonic_lock:
            ret = list(self._ultrasonic_readings)
        return ret

    def is_ultrasonic_healthy(self, stale_threshold_s: float = 0.5) -> bool:
        with self._ultrasonic_lock:
            if self._last_ultrasonic_time == 0:
                return False
            return (time.monotonic() - self._last_ultrasonic_time) <= stale_threshold_s

    def _imu_reader(self) -> None:
        warmup = 0
        while not self._imu_stop.is_set():
            try:
                if self._serial and self._serial.in_waiting > 0:
                    line = self._serial.readline().decode('ascii', errors='ignore').strip()
                    warmup += 1
                    if warmup <= 10:
                        continue
                    heading = float(line)
                    if 0.0 <= heading < 360.0:
                        with self._heading_lock:
                            if self._imu_initial_heading is None:
                                self._imu_initial_heading = heading
                            self._current_heading = (heading - self._imu_initial_heading) % 360
                            self._last_heading_time = time.monotonic()
                        self._imu_ready.set()
                        self._sample_n += 1
                else:
                    time.sleep(0.005)
            except serial.SerialException as e:
                print(f"[HAL] Serial error: {e}")
                time.sleep(0.5)
            except ValueError:
                        pass
            except Exception as e:
                print(f"[HAL] Unexpected error: {e}")

    @staticmethod
    def _find_imu_port(preferred: str = _IMU_PORT) -> str:
        try:
            from serial.tools.list_ports import comports
            for info in comports():
                if info.vid in _ARDUINO_VIDS:
                    print(f"[HAL] Auto-discovered port: {info.device} (VID=0x{info.vid:04X} '{info.description}')")
                    return info.device
        except Exception as e:
            print(f"[HAL] Port auto-discovery failed: {e}")

        print(f"[HAL] Using fallback port: {preferred}")
        return preferred

    def start_imu(self, port: Optional[str] = None, baud: int = _IMU_BAUD, timeout_s: int = 10) -> bool:
        if port is None:
            port = self._find_imu_port(_IMU_PORT)
        try:
            self._serial = serial.Serial(port, baud, timeout=1)
        except Exception as e:
            print(f"[HAL] Could not open {port}: {e}")
            return False

        self._imu_stop.clear()
        self._imu_thread = threading.Thread(
            target=self._imu_reader, daemon=True, name="imu-reader"
        )
        self._imu_thread.start()

        print("[HAL] Waiting for first IMU reading...")
        if not self._imu_ready.wait(timeout=timeout_s):
            print("[HAL] No heading received within timeout")
            return False

        time.sleep(0.5)
        print(f"[HAL] Ready - heading: {self.get_heading():.1f} deg")
        return True

    def stop_imu(self) -> None:
        self._imu_stop.set()
        if self._imu_thread:
            self._imu_thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def is_imu_healthy(self, stale_threshold_s: float = 0.5) -> bool:
        with self._heading_lock:
            if self._last_heading_time == 0:
                return False
            return (time.monotonic() - self._last_heading_time) <= stale_threshold_s

    def get_heading(self) -> float:
        with self._heading_lock:
            if self._last_heading_time == 0.0:
                raise RuntimeError("[HAL] No heading received - is start_imu() complete?")
            age = time.monotonic() - self._last_heading_time
            if age > 0.5:
                raise RuntimeError(
                    f"[HAL] Heading is {age:.1f}s stale - IMU connection may be lost"
                )
            return self._current_heading
        
    def set_imu_offset(self, heading_offset: float) -> None:
        self._imu_initial_heading = heading_offset

    def recalibrate_heading(self, known_heading: float) -> None:
        with self._heading_lock:
            raw = (self._current_heading + self._imu_initial_heading) % 360
            self._imu_initial_heading = (raw - known_heading) % 360

_default = Robot()

def read_front_left_encoder() -> int:
    return _default.read_front_left_encoder()

def read_back_right_encoder() -> int:
    return _default.read_back_right_encoder()

def init() -> None:
    _default.init()

def cleanup() -> None:
    _default.cleanup()

def stop_motors() -> None:
    _default.stop_motors()

def set_individual_duties(left_direction: int, left_duty: float,
                          right_direction: int, right_duty: float) -> None:
    _default.set_individual_duties(left_direction, left_duty, right_direction, right_duty)

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

def start_ultrasonic() -> bool:
    return _default.start_ultrasonic()

def stop_ultrasonic() -> None:
    _default.stop_ultrasonic()

def is_ultrasonic_healthy() -> bool:
    return _default.is_ultrasonic_healthy()

def get_ultrasonic_distance() -> float:
    return _default.get_ultrasonic_distance()

def get_recent_ultrasonic_readings() -> list:
    return _default.get_recent_ultrasonic_readings()

def start_imu(port: Optional[str] = None, baud: int = _IMU_BAUD, timeout_s: int = 10) -> bool:
    return _default.start_imu(port, baud, timeout_s)

def stop_imu() -> None:
    _default.stop_imu()

def is_imu_healthy(stale_threshold_s: float = 0.5) -> bool:
    return _default.is_imu_healthy(stale_threshold_s)

def get_heading() -> float:
    return _default.get_heading()

def set_imu_offset(heading_offset: float) -> None:
    return _default.set_imu_offset(heading_offset)

def recalibrate_heading(known_heading: float) -> None:
    return _default.recalibrate_heading(known_heading)
