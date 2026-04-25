import logging
import queue
import threading
import time
from typing import Optional

from fp import hal
from fp import navigation
from fp.fsm.context import RobotContext, FSMTestHooks
from fp.fsm.states import State, _STATE_CLASSES, _pose

logger = logging.getLogger(__name__)


# REPL control bus

class FsmControl:
    """Shared control object between Runner and REPL."""

    def __init__(self):
        self.pause_event = threading.Event()
        self.resume_event = threading.Event()
        self.paused = threading.Event()
        self.force_transition_queue: queue.Queue = queue.Queue()
        self.step_event = threading.Event()          # single-step while paused
        self.current_state: Optional[State] = None  # updated by runner each tick


# Runner

class Runner:
    def run(
        self,
        ctx: RobotContext,
        hooks: Optional[FSMTestHooks] = None,
        control: Optional[FsmControl] = None,
    ) -> None:
        ctx.hooks = hooks

        state = State.INIT
        state_obj = _STATE_CLASSES[state]
        state_obj.on_enter(ctx)

        logger.info(f"[SM] Starting  zone=({ctx.zone_x}, {ctx.zone_y})  target={ctx.target_color}")

        try:
            while state != State.DONE:
                if control is not None:
                    control.current_state = state

                # REPL pause hook
                if control is not None and control.pause_event.is_set():
                    control.pause_event.clear()
                    control.paused.set()
                    while True:
                        if control.resume_event.is_set():
                            control.resume_event.clear()
                            control.paused.clear()
                            break
                        if control.step_event.is_set():
                            control.step_event.clear()
                            control.paused.clear()
                            control.pause_event.set()  # re-pause after this tick
                            break
                        time.sleep(0.05)

                # REPL force-transition hook
                if control is not None and not control.force_transition_queue.empty():
                    next_state = control.force_transition_queue.get_nowait()
                    state_obj.on_exit(ctx)
                    state = next_state
                    state_obj = _STATE_CLASSES[state]
                    state_obj.on_enter(ctx)
                    continue

                if state not in (State.INIT, State.COMPLETE) and ctx.time_expired():
                    logger.info(f"[SM] Time limit  {_pose()}")
                    state_obj.on_exit(ctx)
                    state = State.COMPLETE
                    state_obj = _STATE_CLASSES[state]
                    state_obj.on_enter(ctx)
                    continue

                logger.info(f"\n{'─' * 60}")
                logger.info(f"[SM] -> {state.name}  {_pose()}")

                next_state = state_obj.step(ctx)

                if next_state != state:
                    state_obj.on_exit(ctx)
                    state = next_state
                    if state == State.DONE:
                        break
                    state_obj = _STATE_CLASSES[state]
                    state_obj.on_enter(ctx)

        except KeyboardInterrupt:
            logger.info(f"\n[SM] Interrupted  {_pose()}")
            hal.stop_motors()
            hal.gripper_open()
        except Exception as exc:  # noqa: BLE001
            logger.info(f"\n[SM] UNHANDLED EXCEPTION: {exc!r}  {_pose()}")
            hal.stop_motors()
            hal.gripper_open()
            raise

        fx, fy, fh = navigation.get_pose()
        logger.info(f"\n{'─' * 60}")
        logger.info(f"[SM] Done - score: {ctx.score}  blocks: {ctx.blocks_delivered}")
        logger.info(f"[SM] Final pose: ({fx:.1f}, {fy:.1f}) cm  {fh:.1f} deg")

        for t in ctx._image_save_threads:
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning(f"[SM] Image-save thread '{t.name}' did not finish within 5 s")


# Public entry point

def run(
    ctx: RobotContext,
    hooks: Optional[FSMTestHooks] = None,
    control: Optional[FsmControl] = None,
) -> None:
    """Run the FSM until DONE.  ctx.picam2 must be set before calling."""
    Runner().run(ctx, hooks=hooks, control=control)
