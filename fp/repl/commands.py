from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fp import navigation, tuning
from fp.repl.tuning_diff import compute_diff, save_to_toml


@dataclass
class CommandEnv:
    ctx: object
    control: object
    session_snapshot: dict
    toml_path: str


_COMMANDS: dict[str, tuple[Callable, str]] = {}


def _cmd(name: str, help_text: str):
    def decorator(fn: Callable) -> Callable:
        _COMMANDS[name] = (fn, help_text)
        return fn
    return decorator


def dispatch(line: str, env: CommandEnv) -> str:
    parts = line.strip().split()
    if not parts:
        return ""
    name = parts[0].lower()
    args = parts[1:]
    entry = _COMMANDS.get(name)
    if entry is None:
        return f"Unknown command: {name!r}  (type 'help' for a list)"
    handler, _ = entry
    try:
        return handler(args, env) or ""
    except Exception as exc:  # noqa: BLE001
        return f"Error in '{name}': {exc}"


# Inspection

@_cmd("pose", "print current robot pose (x, y, heading_deg)")
def _pose(args, env):
    x, y, hdg = navigation.get_pose()
    return f"pose: ({x:.2f}, {y:.2f}) cm  {hdg:.1f} deg"


@_cmd("cov", "print EKF covariance diagonal")
def _cov(args, env):
    try:
        import numpy as np
        diag = np.diag(env.ctx.ekf.get_covariance())
        return "cov diag: [" + ", ".join(f"{v:.4f}" for v in diag) + "]"
    except Exception:
        return "EKF not available"


@_cmd("state", "print current FSM state")
def _state(args, env):
    s = env.control.current_state
    return f"FSM state: {s.name if s is not None else 'not started'}"


@_cmd("blocks", "print scoring and target colour")
def _blocks(args, env):
    ctx = env.ctx
    return (
        f"blocks_delivered: {ctx.blocks_delivered}  "
        f"score: {ctx.score}  "
        f"target: {ctx.target_color}"
    )


# Control

@_cmd("pause", "request FSM pause (takes effect before next state tick)")
def _pause(args, env):
    if env.control.paused.is_set():
        return "FSM is already paused."
    env.control.pause_event.set()
    return "Pause requested."


@_cmd("resume", "resume a paused FSM")
def _resume(args, env):
    if not env.control.paused.is_set():
        return "FSM is not paused."
    env.control.resume_event.set()
    return "Resumed."


@_cmd("step", "execute one FSM tick while paused, then re-pause")
def _step(args, env):
    if not env.control.paused.is_set():
        return "FSM is not paused.  Use 'pause' first."
    env.control.step_event.set()
    return "Step queued."


@_cmd("force", "force FSM to a state by name (e.g.: force DISCOVER)")
def _force(args, env):
    if not args:
        return "Usage: force STATE_NAME"
    from fp.fsm.states import State
    name = args[0].upper()
    try:
        state = State[name]
    except KeyError:
        valid = ", ".join(s.name for s in State if s != State.DONE)
        return f"Unknown state: {name!r}.  Valid: {valid}"
    if state == State.DONE:
        return "Cannot force-transition to DONE."
    env.control.force_transition_queue.put(state)
    return f"Queued -> {state.name}"


# Tuning inspection and mutation

@_cmd("tune", "list all tuning params (* = changed this session)")
def _tune(args, env):
    params = tuning.list_params()
    snap = env.session_snapshot
    lines = []
    for k, (v, typename) in sorted(params.items()):
        marker = "* " if snap.get(k) != v else "  "
        lines.append(f"  {marker}{k}: {v!r}  ({typename})")
    return "Tuning params (* = changed this session):\n" + "\n".join(lines)


@_cmd("set", "set a tuning param: set KEY VALUE")
def _set(args, env):
    if len(args) < 2:
        return "Usage: set KEY VALUE"
    key = args[0].upper()
    raw = args[1]
    try:
        try:
            val = int(raw)
        except ValueError:
            val = float(raw)
    except ValueError:
        return f"Invalid numeric value: {raw!r}"
    try:
        tuning.set_param(key, val)
        return f"{key} = {tuning.get_param(key)!r}"
    except KeyError:
        return f"Unknown param: {key!r}"
    except TypeError as exc:
        return f"Type error: {exc}"


@_cmd("revert", "revert a param to session-start value: revert KEY")
def _revert(args, env):
    if not args:
        return "Usage: revert KEY"
    key = args[0].upper()
    if key not in env.session_snapshot:
        return f"Unknown param: {key!r}"
    val = env.session_snapshot[key]
    tuning.set_param(key, val)
    return f"{key} reverted to {val!r}"


@_cmd(":diff", "show tuning changes made this session")
def _diff(args, env):
    changed = compute_diff(env.session_snapshot)
    if not changed:
        return "No changes this session."
    lines = [f"  {k}: {old!r} -> {new!r}" for k, (old, new) in sorted(changed.items())]
    return "\n".join(lines)


@_cmd(":save", "save session changes to runtime.toml (merged with existing)")
def _save(args, env):
    n = save_to_toml(env.toml_path, env.session_snapshot)
    return f"Saved {n} param(s) to {env.toml_path!r}"


@_cmd(":reset", "revert ALL params to session-start values")
def _reset(args, env):
    for k, v in env.session_snapshot.items():
        tuning.set_param(k, v)
    return f"Reverted all {len(env.session_snapshot)} params to session defaults."


# Help

@_cmd("help", "list available commands")
def _help(args, env):
    lines = ["Available commands:"]
    for name, (_, desc) in sorted(_COMMANDS.items()):
        lines.append(f"  {name:<16}  {desc}")
    lines.append("")
    lines.append("  quit / exit        exit the REPL (FSM keeps running if unpaused)")
    return "\n".join(lines)
