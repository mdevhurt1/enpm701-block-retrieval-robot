"""
fp/repl/tuning_diff.py - Session tuning diff + runtime.toml persistence.

Tracks parameter changes since REPL start and persists them as flat TOML.
The format is intentionally simple (one 'KEY = value' line per param),
human-editable on the robot without a JSON editor — unlike tuning.save()
which produces JSON and is better suited for offline archiving.
"""
from __future__ import annotations

import os

from fp import tuning


def capture_defaults() -> dict:
    """Return a snapshot of current tuning values.

    Call once at REPL start (after any load_toml_if_exists call) so the
    snapshot reflects the starting state for this session.
    """
    return {k: v for k, (v, _) in tuning.list_params().items()}


def compute_diff(snapshot: dict) -> dict:
    """Return {key: (snapshot_val, current_val)} for params changed since snapshot."""
    result = {}
    for k, (curr, _) in tuning.list_params().items():
        snap_val = snapshot.get(k, curr)
        if curr != snap_val:
            result[k] = (snap_val, curr)
    return result


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    s = str(float(v))
    if "." not in s and "e" not in s.lower():
        s += ".0"
    return s


def _read_toml_raw(path: str) -> dict:
    """Read a flat key=value TOML file; return {key: parsed_value}.

    Only handles int and float scalars (all tuning params are numeric).
    Silently skips blank lines, comments, and malformed entries.
    """
    result: dict = {}
    if not os.path.exists(path):
        return result
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            raw = raw.strip()
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    pass
    return result


def save_to_toml(path: str, snapshot: dict) -> int:
    """Persist session changes to *path*, merging with existing file contents.

    Params changed in this session are added/updated.  Params already in the
    file but not changed this session are preserved at their current values.
    Returns the total number of params written to the file.
    """
    existing = _read_toml_raw(path)
    current = {k: v for k, (v, _) in tuning.list_params().items()}
    merged: dict = dict(existing)  # preserve all prior saves
    for key, curr_val in current.items():
        snap_val = snapshot.get(key, curr_val)
        if curr_val != snap_val:
            merged[key] = curr_val  # record session change; existing key unchanged if not modified
    with open(path, "w") as fh:
        for key in sorted(merged):
            fh.write(f"{key} = {_toml_scalar(merged[key])}\n")
    return len(merged)


def load_from_toml(path: str) -> int:
    """Load parameter overrides from *path* and apply via tuning.set_param.

    Returns the number of params successfully applied.
    Silently skips unknown keys and badly-typed values.
    """
    data = _read_toml_raw(path)
    n = 0
    for k, v in data.items():
        try:
            tuning.set_param(k, v)
            n += 1
        except (KeyError, TypeError):
            pass
    return n


def load_toml_if_exists(path: str) -> int:
    """Apply overrides from *path* if it exists; return 0 otherwise."""
    if not os.path.exists(path):
        return 0
    return load_from_toml(path)
