import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import pytest

from fp.cli import _build_parser


class TestCliParser:
    def test_no_subcommand_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.mode is None
        assert args.repl is False
        assert args.no_wait is False
        assert args.color is None

    def test_run_subcommand_sets_mode(self):
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.mode == "run"
        assert args.repl is False

    def test_run_subcommand_with_repl(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--repl"])
        assert args.mode == "run"
        assert args.repl is True

    def test_run_subcommand_with_color(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--color", "red"])
        assert args.mode == "run"
        assert args.color == "red"

    def test_run_subcommand_with_no_wait(self):
        parser = _build_parser()
        args = parser.parse_args(["run", "--no-wait"])
        assert args.mode == "run"
        assert args.no_wait is True

    def test_top_level_repl_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--repl"])
        assert args.mode is None
        assert args.repl is True

    def test_top_level_color_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--color", "blue"])
        assert args.mode is None
        assert args.color == "blue"

    def test_invalid_color_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--color", "purple"])
