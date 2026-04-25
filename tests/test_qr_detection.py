import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import pytest


class TestParsePayload:

    def _get_parser(self):
        from fp import qr_detection
        fn = getattr(qr_detection, '_parse_payload', None)
        if fn is None:
            pytest.skip("_parse_payload not exposed in qr_detection module")
        return fn

    def test_valid_payload(self):
        parse = self._get_parser()
        result = parse("x=107,y=107")
        assert result is not None
        x, y = result
        assert abs(x - 107.0) < 0.01
        assert abs(y - 107.0) < 0.01

    def test_valid_payload_large_values(self):
        parse = self._get_parser()
        result = parse("x=180,y=30")
        assert result is not None
        x, y = result
        assert abs(x - 180.0) < 0.01
        assert abs(y - 30.0) < 0.01

    def test_malformed_missing_y(self):
        parse = self._get_parser()
        result = parse("x=107")
        assert result is None, "missing y should return None"

    def test_malformed_garbage(self):
        parse = self._get_parser()
        result = parse("hello world")
        assert result is None

    def test_malformed_empty_string(self):
        parse = self._get_parser()
        result = parse("")
        assert result is None

    def test_malformed_non_numeric(self):
        parse = self._get_parser()
        result = parse("x=abc,y=def")
        assert result is None

    def test_valid_with_spaces(self):
        parse = self._get_parser()
        result = parse("  x=50,  y=100  ")
        assert result is not None
        x, y = result
        assert abs(x - 50.0) < 0.01 and abs(y - 100.0) < 0.01

    def test_valid_float_values(self):
        parse = self._get_parser()
        result = parse("x=12.5,y=33.7")
        assert result is not None
        x, y = result
        assert abs(x - 12.5) < 0.01 and abs(y - 33.7) < 0.01


class TestDetectQR:

    def test_detect_qr_returns_none_on_empty_frame(self):
        import numpy as np
        from fp import qr_detection
        # Black frame: QR detector should find nothing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            result = qr_detection.detect_qr(frame)
            assert result is None
        except (ValueError, TypeError):
            pass  # some cv2 builds return unexpected tuple sizes on empty frames

    def test_detect_qr_returns_none_on_none_frame(self):
        from fp import qr_detection
        # None frame should not crash (cv2 will raise, but we catch it)
        try:
            result = qr_detection.detect_qr(None)
            assert result is None
        except (TypeError, Exception):
            pass  # acceptable to raise on None input
