from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TOML = "runtime.toml"


def start_repl(
    ctx,
    control,
    picam2=None,
    toml_path: str = _DEFAULT_TOML,
) -> None:
    """Run the FSM in a worker thread and serve the REPL on the calling thread.

    cli.py must have already called load_toml_if_exists() before this so that
    the session snapshot captures the post-load state as the baseline.
    """
    from fp.fsm.runner import run as fsm_run
    from fp.repl.tuning_diff import capture_defaults
    from fp.repl.commands import CommandEnv, dispatch

    snapshot = capture_defaults()
    env = CommandEnv(
        ctx=ctx,
        control=control,
        session_snapshot=snapshot,
        toml_path=toml_path,
    )

    t = threading.Thread(
        target=fsm_run,
        args=(ctx,),
        kwargs={"control": control},
        name="fsm-worker",
        daemon=True,
    )
    t.start()
    logger.info("[REPL] FSM worker thread started")

    _run_input_loop(env, t)

    logger.info("[REPL] Input loop exited; waiting for FSM worker (up to 10 s)")
    t.join(timeout=10.0)
    if t.is_alive():
        logger.warning("[REPL] FSM worker did not finish within 10 s")


def _get_line_fn():
    try:
        from prompt_toolkit import PromptSession
        session = PromptSession()

        def _get(prompt: str) -> str:
            return session.prompt(prompt)

        return _get
    except ImportError:
        return input


def _run_input_loop(env, fsm_thread: threading.Thread) -> None:
    get_line = _get_line_fn()

    print()
    print("  Grand Challenge REPL  —  FSM running in background.")
    print("  Type 'help' for commands.  Ctrl-D or 'quit' to exit.")
    print()

    while True:
        try:
            line = get_line("fp> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()

        if not line:
            if not fsm_thread.is_alive():
                print("[REPL] FSM thread has completed.")
                break
            continue

        if line.lower() in ("quit", "exit", "q"):
            break

        result = dispatch(line, env)
        if result:
            print(result)

        if not fsm_thread.is_alive():
            print("[REPL] FSM thread has completed.")
            break
