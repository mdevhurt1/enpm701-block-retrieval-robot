import sys
from unittest.mock import MagicMock, patch

# Mock RPi.GPIO
gpio_mock = MagicMock()
gpio_mock.BOARD = 10
gpio_mock.IN    = 1
gpio_mock.OUT   = 0
gpio_mock.BOTH  = 3
gpio_mock.PUD_UP = 22
gpio_mock.input.return_value = 0
sys.modules['RPi']        = MagicMock()
sys.modules['RPi.GPIO']   = gpio_mock

# Mock serial (pyserial)
serial_mock = MagicMock()
serial_mock.Serial = MagicMock()
serial_mock.tools  = MagicMock()
serial_mock.tools.list_ports = MagicMock()
serial_mock.tools.list_ports.comports.return_value = []
sys.modules['serial']                    = serial_mock
sys.modules['serial.tools']              = serial_mock.tools
sys.modules['serial.tools.list_ports']   = serial_mock.tools.list_ports

# Mock picamera2
picam_mock = MagicMock()
picam_mock.Picamera2 = MagicMock()
sys.modules['picamera2'] = picam_mock

# Mock libcamera (Transform)
sys.modules['libcamera'] = MagicMock()

# Mock cv2 (only if not already installed)
try:
    import cv2 as _cv2_real  # noqa: F401
except ImportError:
    import numpy as _np_stub
    cv2_mock = MagicMock()
    cv2_mock.cvtColor.return_value = _np_stub.zeros((480, 640, 3), dtype=_np_stub.uint8)
    cv2_mock.COLOR_BGR2HSV = 40
    cv2_mock.inRange.return_value = _np_stub.zeros((480, 640), dtype=_np_stub.uint8)
    cv2_mock.findContours.return_value = ([], None)
    cv2_mock.minAreaRect.return_value = ((0, 0), (1, 1), 0)
    sys.modules['cv2'] = cv2_mock
