import sys
from unittest.mock import MagicMock

sys.modules.setdefault('pigpio', MagicMock())

_gpio_mod = MagicMock()
_rpi_mod = MagicMock()
_rpi_mod.GPIO = _gpio_mod
sys.modules.setdefault('RPi', _rpi_mod)
sys.modules.setdefault('RPi.GPIO', _gpio_mod)

sys.modules.setdefault('serial', MagicMock())
sys.modules.setdefault('serial.tools', MagicMock())
sys.modules.setdefault('serial.tools.list_ports', MagicMock())

sys.modules.setdefault('picamera2', MagicMock())
sys.modules.setdefault('libcamera', MagicMock())
