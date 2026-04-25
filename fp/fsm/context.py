import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fp import tuning


# Sub-dataclasses

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    zone_x: float = 0.0
    zone_y: float = 0.0


@dataclass
class Scoring:
    score: int = 0
    blocks_delivered: int = 0
    next_color_index: int = 0
    color_override: Optional[str] = None


@dataclass
class Search:
    sweep_index: int = 0
    sweep_count: int = 0
    home_heading: float = 0.0
    current_path: List = field(default_factory=list)


@dataclass
class Cooldowns:
    last_update_time: float = 0.0
    qr_last_fix_time: float = 0.0
    wall_cooldown: Dict[str, float] = field(default_factory=dict)


# Test hooks

@dataclass
class FSMTestHooks:
    """Optional test-time overrides passed to run(); not part of RobotContext."""
    skip_init_calibration: bool = False
    stop_after_delivery: bool = False


# Configuration referenced by target_color

_COLOR_ORDER = ["red", "green", "blue"]


# RobotContext

class RobotContext:

    def __init__(self):
        self.pose = Pose()
        self.scoring = Scoring()
        self.search = Search()
        self.cooldowns = Cooldowns()

        # Camera handle (must be set before run())
        self.picam2 = None

        # State to resume after AVOID / UPDATE
        self.return_state = None  # set to State.DISCOVER by runner before use

        # EKF localization (created in INIT)
        self.ekf = None

        # Occupancy grid planner (created in INIT)
        self.planner = None

        # Timing
        self.run_start: float = 0.0
        self.run_limit_s: float = tuning.RUN_LIMIT_S

        # Tracked background threads joined before run() returns
        self._image_save_threads: List = []

        # Approach failure guard
        self.approach_fail_count: int = 0

        # Test hooks injected by runner; None during autonomous operation
        self.hooks: Optional[FSMTestHooks] = None


    @property
    def x(self) -> float:
        return self.pose.x

    @x.setter
    def x(self, v: float) -> None:
        self.pose.x = v

    @property
    def y(self) -> float:
        return self.pose.y

    @y.setter
    def y(self, v: float) -> None:
        self.pose.y = v

    @property
    def zone_x(self) -> float:
        return self.pose.zone_x

    @zone_x.setter
    def zone_x(self, v: float) -> None:
        self.pose.zone_x = v

    @property
    def zone_y(self) -> float:
        return self.pose.zone_y

    @zone_y.setter
    def zone_y(self, v: float) -> None:
        self.pose.zone_y = v


    @property
    def score(self) -> int:
        return self.scoring.score

    @score.setter
    def score(self, v: int) -> None:
        self.scoring.score = v

    @property
    def blocks_delivered(self) -> int:
        return self.scoring.blocks_delivered

    @blocks_delivered.setter
    def blocks_delivered(self, v: int) -> None:
        self.scoring.blocks_delivered = v

    @property
    def next_color_index(self) -> int:
        return self.scoring.next_color_index

    @next_color_index.setter
    def next_color_index(self, v: int) -> None:
        self.scoring.next_color_index = v

    @property
    def color_override(self) -> Optional[str]:
        return self.scoring.color_override

    @color_override.setter
    def color_override(self, v: Optional[str]) -> None:
        self.scoring.color_override = v


    @property
    def sweep_index(self) -> int:
        return self.search.sweep_index

    @sweep_index.setter
    def sweep_index(self, v: int) -> None:
        self.search.sweep_index = v

    @property
    def sweep_count(self) -> int:
        return self.search.sweep_count

    @sweep_count.setter
    def sweep_count(self, v: int) -> None:
        self.search.sweep_count = v

    @property
    def home_heading(self) -> float:
        return self.search.home_heading

    @home_heading.setter
    def home_heading(self, v: float) -> None:
        self.search.home_heading = v

    @property
    def current_path(self) -> List:
        return self.search.current_path

    @current_path.setter
    def current_path(self, v: List) -> None:
        self.search.current_path = v


    @property
    def last_update_time(self) -> float:
        return self.cooldowns.last_update_time

    @last_update_time.setter
    def last_update_time(self, v: float) -> None:
        self.cooldowns.last_update_time = v

    @property
    def qr_last_fix_time(self) -> float:
        return self.cooldowns.qr_last_fix_time

    @qr_last_fix_time.setter
    def qr_last_fix_time(self, v: float) -> None:
        self.cooldowns.qr_last_fix_time = v

    @property
    def _wall_update_cooldown(self) -> Dict[str, float]:
        return self.cooldowns.wall_cooldown

    @_wall_update_cooldown.setter
    def _wall_update_cooldown(self, v: Dict[str, float]) -> None:
        self.cooldowns.wall_cooldown = v


    @property
    def target_color(self) -> str:
        if self.scoring.color_override:
            return self.scoring.color_override
        return _COLOR_ORDER[self.scoring.next_color_index % len(_COLOR_ORDER)]

    def time_expired(self) -> bool:
        return (_time.monotonic() - self.run_start) >= self.run_limit_s


# Helpers

def set_collection_zone(ctx: RobotContext, x: float, y: float) -> None:
    """Override the collection zone centre (cm from SW corner of arena)."""
    import logging
    ctx.zone_x = round(x, 1)
    ctx.zone_y = round(y, 1)
    logging.getLogger(__name__).info(
        f"[SM] Collection zone set to ({ctx.zone_x}, {ctx.zone_y}) cm"
    )
