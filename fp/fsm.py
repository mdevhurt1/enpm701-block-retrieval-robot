# Final Project - Mission State Machine
# Seven-state FSM (LOCALIZE, SEARCH, APPROACH, GRAB, CARRY_HOME, DROP,
# STEP_OUT) that autonomously collects colored blocks in color-order
# (red, green, blue) and delivers them to the NW collection zone.
#
# Requirements: fp.arena, fp.motion, fp.search, fp.vision, fp.localization
import logging
import time
from enum import Enum

from fp import arena
from fp import motion
from fp import search
from fp import vision
from fp.localization import Localizer
from fp.motion import TargetDetected, ApproachVisualStuck

logger = logging.getLogger(__name__)

# mission parameters
TOTAL_BLOCKS = 9
COLOR_ORDER = ("red", "green", "blue")
T_HARD_S = 600.0
COLLECTION_ZONE_MARGIN = 10.0
# Below this get_search_quality value we treat the cell as already-quality-
# searched and skip scan_360. 100 is the value get_search_quality returns
# for a cell at the global minimum visit count, so this threshold means
# "scan only the least-visited cells in the arena". Tuneable; lower values
# scan more aggressively, higher values skip more aggressively.
SEARCH_QUALITY_THRESHOLD = 100.0


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
            # When carrying a block, refresh the position estimate before the
            # long carry leg. Search/approach can leave us with stale odometry
            # (the SE->NW carry-home observed landing at the centre of the
            # north wall is the failure mode this addresses); without a
            # pre-flight localize the entire 2 m drive home runs on whatever
            # estimate the approach finished with. bypass_cooldown overrides
            # the 30 s gate so this fires even if a localize happened during
            # the approach loop.
            if self.holding_target:
                logger.info("[FSM] LOCALIZE pre-carry localize (bypass_cooldown)")
                pre_carry_loc = motion.localize(self.localizer, bypass_cooldown=True)
                logger.info(f"[FSM] LOCALIZE pre-carry localize {'ok' if pre_carry_loc else 'failed'}, pose=({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")

            px, py = arena.COLLECTION_ZONE_CENTER
            logger.info(f"[FSM] LOCALIZE pos=({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f}) target: ({px}, {py})")
            move = motion.move_to_point(self.localizer, px, py, target=self._target_color(), carrying_block=self.holding_target)
            if move:
                logger.info("[FSM] LOCALIZE move succeeded")
            else:
                logger.info("[FSM] LOCALIZE move failed")
            
            if self.holding_target:
                logger.info("[FSM] LOCALIZE post-carry localize (bypass_cooldown)")
                pre_carry_loc = motion.localize(self.localizer, bypass_cooldown=True)
                logger.info(f"[FSM] LOCALIZE post-carry localize {'ok' if pre_carry_loc else 'failed'}, pose=({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
                ret = State.CARRY_HOME
            else:
                ret = State.SEARCH

            localize = motion.localize(self.localizer)
            if localize:
                logger.info(f"[FSM] LOCALIZE completed, new position ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            else:
                logger.info(f"[FSM] LOCALIZE failed, position ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            
            return ret
        except TargetDetected as e:
            logger.info(f"[FSM] LOCALIZE target detected during localization {e.detection.area:.1f}")
            return State.APPROACH

    def _on_search(self) -> State:
        target = self._target_color()
        try:
            pose_x = self.localizer.pose.x_cm
            pose_y = self.localizer.pose.y_cm
            logger.info(f"[FSM] SEARCH starting from ({pose_x:.1f}, {pose_y:.1f}) targeting {target}")

            # Skip the ~22 s scan_360 when this cell is already low-quality
            # (heavily visited) AND has no cached sighting *of the current
            # target color*. A target-color sighting forces a re-scan even
            # on a spent cell - the block we previously saw may still be
            # there. Sightings of other colors don't override; they aren't
            # informative for the current search.
            quality = self.planner.get_search_quality(pose_x, pose_y)
            has_sighting = self.planner.has_sighting_in_cell(pose_x, pose_y, target)
            if quality < SEARCH_QUALITY_THRESHOLD and not has_sighting:
                logger.info(f"[FSM] SEARCH skipping scan at ({pose_x:.1f}, {pose_y:.1f}); "
                            f"quality={quality:.0f} has_sighting={has_sighting}")
                self.planner.mark_visited(pose_x, pose_y)
            else:
                skip_zone = self.delivered >= len(COLOR_ORDER)
                det = self.planner.scan_360(self.localizer, target, skip_collection_zone=skip_zone)
                if det is not None:
                    logger.info(f"[FSM] SEARCH found {target} -> APPROACH")
                    return State.APPROACH

            logger.info(f"[FSM] SEARCH did not find {target}; moving to next waypoint")
            # Prefer cached non-target sightings: scan_360 records (rx, ry,
            # heading) every time it spots a non-target color, so when we
            # later need that color we can revisit the position we last saw
            # it from instead of grinding through the lawnmower grid. Pop
            # consumes the sighting so a stale entry won't loop us.
            sighting = self.planner.get_sighting(target)
            if sighting is not None:
                x, y, _ = sighting
                logger.info(f"[FSM] SEARCH using cached sighting of {target} at ({x:.1f}, {y:.1f})")
            else:
                x, y = self.planner.next_pose(self.localizer.pose.x_cm, self.localizer.pose.y_cm)

            logger.info(f"[FSM] SEARCH moving to ({x:.1f}, {y:.1f})")
            move = motion.move_to_point(self.localizer, x, y, target=target, navigation_retries=0)
            if move:
                logger.info(f"[FSM] SEARCH reached ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            else:
                logger.info(f"[FSM] SEARCH did not reach ({x:.1f}, {y:.1f}), current position ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
                self.planner.mark_visited(x, y)

            return State.SEARCH
        except TargetDetected as e:
            logger.info(f"[FSM] SEARCH target detected during search {e.detection.area:.1f}")
            return State.APPROACH

    def _on_approach(self) -> State:
        target = self._target_color()
        try:
            logger.info(f"[FSM] APPROACH starting from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}) approaching {target} block visually")
            approach = motion.approach_visual(self.localizer, target)
            if approach:
                logger.info(f"[FSM] APPROACH {target} block in reach -> GRAB")
                return State.GRAB
            else:
                logger.info(f"[FSM] APPROACH failed; returning to search")
                return State.SEARCH
        except TargetDetected as e:
            logger.info(f"[FSM] APPROACH exception fired for target block {e.detection.area:.1f}")
            return State.GRAB
        except ApproachVisualStuck:
            logger.info(f"[FSM] APPROACH stuck; returning to localize")
            return State.LOCALIZE

    def _on_grab(self) -> State:
        target = self._target_color()

        logger.info(f"[FSM] GRAB attempting to grab {target} at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
        grabbed = motion.grab_block(self.localizer, self._target_color())
        if grabbed:
            logger.info(f"[FSM] GRAB successfully grabbed {target} block")
            self.holding_target = True
            return State.LOCALIZE
        else:
            logger.info(f"[FSM] GRAB failed; returning to search")        
            return State.SEARCH
        

    def _on_carry_home(self) -> State:
        target = self._target_color()
        
        cx, cy = arena.COLLECTION_ZONE_CENTER
        logger.info(f"[FSM] CARRY_HOME carrying {target} block to ({cx:.1f}, {cy:.1f}) from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
        move = motion.move_to_point(self.localizer, cx, cy, target=self._target_color(), carrying_block=self.holding_target)
        if move:
            logger.info(f"[FSM] CARRY_HOME successfully arrived at ({cx:.1f}, {cy:.1f})")
        else:
            logger.info(f"[FSM] CARRY_HOME failed to reach ({cx:.1f}, {cy:.1f}), currently at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")    
        
        return State.DROP

    def _on_drop(self) -> State:
        target = self._target_color()
        in_zone = arena.in_collection_zone(self.localizer.pose.x_cm + COLLECTION_ZONE_MARGIN,
                                           self.localizer.pose.y_cm - COLLECTION_ZONE_MARGIN)
        if in_zone:
            logger.info(f"[FSM] DROP dropping {target} block in zone at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
            self.delivered += 1
        else:
            logger.info(f"[FSM] DROP dropping {target} block at ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f})")
        
        drop = motion.release_block(self.localizer)
        self.holding_target = False
        if drop:
            logger.info(f"[FSM] DROP successfully released block")
        else:
            logger.info(f"[FSM] DROP failed to release block")

        logger.info(f"[FSM] delivered {self.delivered} / {self.max_deliveries}")        
        if self.delivered >= self.max_deliveries:
            return State.COMPLETE
        
        return State.STEP_OUT

    def _on_step_out(self) -> State:
        logger.info(f"[FSM] STEP_OUT stepping out from ({self.localizer.pose.x_cm:.1f}, {self.localizer.pose.y_cm:.1f}, {self.localizer.pose.heading:.1f}) reversing")
        pivot = motion.pivot_to_heading(self.localizer, 315)
        move = motion.move_reverse(self.localizer)
        if move:
            logger.info("[FSM] STEP_OUT successful")
        else:
            logger.info("[FSM] STEP_OUT failed")

        return State.SEARCH
    
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
            logger.info("[FSM] step has no handler to call")
            return False
        next_state = handler()
        self.state = next_state
        return next_state not in (State.COMPLETE, State.TIMEOUT)

    def run(self) -> tuple:
        while self.step():
            vision.capture_frame()
        return self.state, self.delivered


def run(loc: Localizer, max_deliveries: int = TOTAL_BLOCKS) -> tuple:
    fsm = FSM(loc, time.monotonic(), max_deliveries=max_deliveries)
    return fsm.run()
