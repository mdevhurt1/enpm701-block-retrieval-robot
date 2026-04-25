import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

import pytest
from fp import tuning


class TestGetParam:
    def test_returns_current_module_value(self):
        assert tuning.get_param('ARRIVAL_TOLERANCE_CM') == tuning.ARRIVAL_TOLERANCE_CM

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown tuning parameter"):
            tuning.get_param('DOES_NOT_EXIST')


class TestSetParam:
    def test_updates_module_attribute(self):
        original = tuning.ARRIVAL_TOLERANCE_CM
        try:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', 5.0)
            assert tuning.ARRIVAL_TOLERANCE_CM == 5.0
        finally:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original)

    def test_get_param_reflects_new_value(self):
        original = tuning.ARRIVAL_TOLERANCE_CM
        try:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', 12.5)
            assert tuning.get_param('ARRIVAL_TOLERANCE_CM') == 12.5
        finally:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original)

    def test_int_param_coerces_float_input(self):
        original = tuning.WALL_SWEEP_SAMPLES
        try:
            tuning.set_param('WALL_SWEEP_SAMPLES', 9.0)
            assert tuning.WALL_SWEEP_SAMPLES == 9
            assert isinstance(tuning.WALL_SWEEP_SAMPLES, int)
        finally:
            tuning.set_param('WALL_SWEEP_SAMPLES', original)

    def test_float_param_coerces_int_input(self):
        original = tuning.ARRIVAL_TOLERANCE_CM
        try:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', 10)
            assert tuning.ARRIVAL_TOLERANCE_CM == 10.0
            assert isinstance(tuning.ARRIVAL_TOLERANCE_CM, float)
        finally:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original)

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            tuning.set_param('ARRIVAL_TOLERANCE_CM', 'banana')

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown tuning parameter"):
            tuning.set_param('NONEXISTENT_PARAM', 1.0)


class TestListParams:
    def test_returns_dict_with_known_params(self):
        params = tuning.list_params()
        assert isinstance(params, dict)
        for key in ('ARRIVAL_TOLERANCE_CM', 'WALL_SWEEP_TRIGGER_CM', 'RUN_LIMIT_S',
                    'WALL_SWEEP_SAMPLES', 'INIT_NORTH_CHECK_BATCHES'):
            assert key in params, f"{key} missing from list_params()"

    def test_each_entry_is_value_and_type_name(self):
        params = tuning.list_params()
        val, type_name = params['ARRIVAL_TOLERANCE_CM']
        assert isinstance(val, float)
        assert type_name == 'float'

    def test_covers_all_registered_params(self):
        params = tuning.list_params()
        assert len(params) >= 17  # at least as many as defined in tuning.py


class TestSaveLoad:
    def test_save_creates_json_file(self, tmp_path):
        path = str(tmp_path / "tuning.json")
        tuning.save(path)
        import json
        with open(path) as fh:
            data = json.load(fh)
        assert 'ARRIVAL_TOLERANCE_CM' in data
        assert data['ARRIVAL_TOLERANCE_CM'] == tuning.ARRIVAL_TOLERANCE_CM

    def test_load_applies_overrides(self, tmp_path):
        import json
        path = str(tmp_path / "override.json")
        original = tuning.ARRIVAL_TOLERANCE_CM
        with open(path, 'w') as fh:
            json.dump({'ARRIVAL_TOLERANCE_CM': 3.5}, fh)
        try:
            n = tuning.load(path)
            assert n == 1
            assert tuning.ARRIVAL_TOLERANCE_CM == 3.5
        finally:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original)

    def test_load_returns_zero_for_missing_file(self, tmp_path):
        n = tuning.load(str(tmp_path / "does_not_exist.json"))
        assert n == 0

    def test_load_skips_unknown_keys(self, tmp_path):
        import json
        path = str(tmp_path / "bad_keys.json")
        with open(path, 'w') as fh:
            json.dump({'UNKNOWN_KEY': 99.0, 'ARRIVAL_TOLERANCE_CM': 7.0}, fh)
        original = tuning.ARRIVAL_TOLERANCE_CM
        try:
            n = tuning.load(path)
            assert n == 1  # only ARRIVAL_TOLERANCE_CM applied
            assert tuning.ARRIVAL_TOLERANCE_CM == 7.0
        finally:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original)

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "roundtrip.json")
        original_tol = tuning.ARRIVAL_TOLERANCE_CM
        original_samples = tuning.WALL_SWEEP_SAMPLES
        try:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', 6.25)
            tuning.set_param('WALL_SWEEP_SAMPLES', 11)
            tuning.save(path)
            # Reset to originals
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original_tol)
            tuning.set_param('WALL_SWEEP_SAMPLES', original_samples)
            # Load and verify
            n = tuning.load(path)
            assert n >= 2
            assert tuning.ARRIVAL_TOLERANCE_CM == 6.25
            assert tuning.WALL_SWEEP_SAMPLES == 11
        finally:
            tuning.set_param('ARRIVAL_TOLERANCE_CM', original_tol)
            tuning.set_param('WALL_SWEEP_SAMPLES', original_samples)
