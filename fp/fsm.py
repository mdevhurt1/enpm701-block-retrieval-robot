import time
from enum import Enum

from fp import arena
from fp import motion
from fp import search
from fp.localization import Localizer
from fp.motion import TargetDetected, ApproachVisualStuck

# mission parameters
TOTAL_BLOCKS = 9
COLOR_ORDER = ("red", "green", "blue")
T_HARD_S = 570.0


class State(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    GRAB = "grab"
    CARRY_HOME = "carry_home"
    DROP = "drop"
    STEP_OUT = "step_out"
    COMPLETE = "complete"
    TIMEOUT = "timeout"
    LOCALIZE = "localize"

class FSM:
    def __init__(self, localizer: Localizer, start_time,
                 max_deliveries: int = TOTAL_BLOCKS) -> None:
        self.localizer = localizer
        self.max_deliveries = max_deliveries
        self.start_time = start_time
        self.state = State.LOCALIZE
        self.delivered = 0
        self.holding_target = False
        self.planner = search.SearchPlanner()
        self._handlers = {
            State.SEARCH: self._on_search,
            State.APPROACH: self._on_approach,
            State.GRAB: self._on_grab,
            State.CARRY_HOME: self._on_carry_home,
            State.DROP: self._on_drop,
            State.STEP_OUT: self._on_step_out,
            State.LOCALIZE: self._on_localize,
        }

    def _on_localize(self) -> State:
        ret = State.SEARCH
        try:
            px, py = arena.COLLECTION_ZONE_CENTER
            print(f"[FSM] LOCALIZE pos=({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f}) target: ({px}, {py})")
            move = motion.move_to_point(self.localizer, px, py, target=self._target_color(), carrying_block=self.holding_target)
            if move:
                print("[FSM] LOCALIZE move succeeded")
            else:
                print("[FSM] LOCALIZE move failed")
            
            if self.holding_target:
                ret = State.CARRY_HOME
            else:
                ret = State.SEARCH

            localize = motion.localize(self.localizer)
            if localize:
                print(f"[FSM] LOCALIZE completed, new position ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            else:
                print(f"[FSM] LOCALIZE failed, position ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            
            return ret
        except TargetDetected as e:
            print(f"[FSM] LOCALIZE target detected during localization {e.detection.area:.1f}")
            return State.APPROACH

    def _on_search(self) -> State:
        target = self._target_color()
        try:
            print(f"[FSM] SEARCH starting from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}) targeting {target}")
            skip_zone = self.delivered >= len(COLOR_ORDER)
            det = self.planner.scan_360(self.localizer, target, skip_collection_zone=skip_zone)
            if det is not None:
                print(f"[FSM] SEARCH found {target} -> APPROACH")
                return State.APPROACH

            print(f"[FSM] SEARCH did not find {target}; moving to next waypoint")
            x, y = self.planner.next_pose(self.localizer.pose.x_cm, self.localizer.pose.y_cm)

            print(f"[FSM] SEARCH moving to ({x:.1f}, {y:.1f})")
            move = motion.move_to_point(self.localizer, x, y, target=target, navigation_retries=0)
            if move:
                print(f"[FSM] SEARCH reached ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            else:
                print(f"[FSM] SEARCH did not reach ({x:.1f}, {y:.1f}), current position ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
                self.planner.mark_visited(x, y)

            return State.SEARCH
        except TargetDetected as e:
            print(f"[FSM] SEARCH target detected during dearch {e.detection.area:.1f}")
            return State.APPROACH

    def _on_approach(self) -> State:
        target = self._target_color()
        try:
            print(f"[FSM] APPROACH starting from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}) approaching {target} block visually")
            approach = motion.approach_visual(self.localizer, target)
            if approach:
                print(f"[FSM] APPROACH {target} block in reach -> GRAB")
                return State.GRAB
            else:
                print(f"[FSM] APPROACH failed; returning to search")
                return State.SEARCH
        except TargetDetected as e:
            print(f"[FSM] APPROACH exception fired for target block {e.detection.area:.1f}")
            return State.GRAB
        except ApproachVisualStuck:
            print(f"[FSM] APPROACH stuck; returning to localize")
            return State.LOCALIZE

    def _on_grab(self) -> State:
        target = self._target_color()

        print(f"[FSM] GRAB attempting to grab {target} at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
        grabbed = motion.grab_block(self.localizer, self._target_color())
        if grabbed:
            print(f"[FSM] GRAB successfully grabbed {target} block")
            self.holding_target = True
            return State.LOCALIZE
        else:
            print(f"[FSM] GRAB failed; returning to search")        
            return State.SEARCH
        

    def _on_carry_home(self) -> State:
        target = self._target_color()
        
        cx, cy = arena.COLLECTION_ZONE_CENTER
        print(f"[FSM] CARRY_HOME carrying {target} block to ({cx:.1f}, {cy:.1f}) from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
        move = motion.move_to_point(self.localizer, cx, cy, target=self._target_color(), carrying_block=self.holding_target)
        if move:
            print(f"[FSM] CARRY_HOME successfully arrived at ({cx:.1f}, {cy:.1f})")
        else:
            print(f"[FSM] CARRY_HOME failed to reach ({cx:.1f}, {cy:.1f}), currently at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")    
        
        return State.DROP

    def _on_drop(self) -> State:
        target = self._target_color()
        in_zone = arena.in_collection_zone(self.localizer.pose.x_cm,
                                           self.localizer.pose.y_cm)
        if in_zone:
            print(f"[FSM] DROP dropping {target} block in zone at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            self.delivered += 1
        else:
            print(f"[FSM] DROP dropping {target} block at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
        
        drop = motion.release_block(self.localizer)
        self.holding_target = False
        if drop:
            print(f"[FSM] DROP successfully released block")
        else:
            print(f"[FSM] DROP failed to release block")

        print(f"[FSM] delivered {self.delivered} / {self.max_deliveries}")        
        if self.delivered >= self.max_deliveries:
            return State.COMPLETE
        
        return State.STEP_OUT

    def _on_step_out(self) -> State:
        print(f"[FSM] STEP_OUT stepping out from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f}) reversing")
        pivot = motion.pivot_to_heading(self.localizer, 315)
        move = motion.move_reverse(self.localizer)
        if move:
            print("[FSM] STEP_OUT successful")
        else:
            print("[FSM] STEP_OUT failed")

        return State.LOCALIZE
    
    def _elapsed_s(self) -> float:
        return time.monotonic() - self.start_time

    def _target_color(self) -> str:
        return COLOR_ORDER[self.delivered % len(COLOR_ORDER)]

    def step(self) -> bool:
        # Hard timeout - applies to all states
        if self._elapsed_s() > T_HARD_S:
            self.state = State.TIMEOUT
            return False

        if self.delivered >= self.max_deliveries:
            self.state = State.COMPLETE
            return False

        handler = self._handlers.get(self.state)
        if handler is None:
            print("[FSM] step has no handler to call")
            return False
        next_state = handler()
        self.state = next_state
        return next_state not in (State.COMPLETE, State.TIMEOUT)

    def run(self) -> tuple:
        while self.step():
            pass
        return self.state, self.delivered


def run(loc: Localizer, max_deliveries: int = TOTAL_BLOCKS) -> tuple:
    fsm = FSM(loc, time.monotonic(), max_deliveries=max_deliveries)
    return fsm.run()
