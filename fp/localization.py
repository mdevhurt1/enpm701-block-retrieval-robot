# Final Project - Pose Tracking and Localizer
# Pose dataclass holds (x_cm, y_cm, heading). Localizer applies encoder
# and ultrasonic motion updates, wall-clearance checks, and path recording.
#
# Requirements: math, dataclasses, fp.arena
from dataclasses import dataclass
import math

from fp import arena

# Robot geometry
ROBOT_HALF_EXTENT_CM    = 20.0
ROBOT_SENSOR_OFFSET_CM  = 15.0   # front sensor is 15 cm forward of robot center.
WALL_CLEARANCE_CM       = 15.0

@dataclass
class Pose:
    x_cm: float = 0.0
    y_cm: float = 0.0
    heading: float = 0.0


class Localizer:
    def __init__(self) -> None:
        self.pose = Pose()
        self.tracker = None

    def set_position(self, x_cm: float, y_cm: float) -> None:
        self.pose.x_cm = x_cm
        self.pose.y_cm = y_cm

    def set_heading(self, heading: float) -> None:
        self.pose.heading = heading

    def apply_forward(self, distance_cm: float, heading: float) -> None:
        x, y = arena.apply_motion(
            self.pose.x_cm, self.pose.y_cm,
            distance_cm, heading
        )
        self.pose.x_cm = x
        self.pose.y_cm = y
        self.pose.heading = heading
        
        if self.tracker is not None:
            self.tracker.record(self.pose.x_cm, self.pose.y_cm)

    def apply_reverse(self, distance_cm: float, heading: float) -> None:
        self.apply_forward(-distance_cm, heading)
    
    def ultrasonic_to_robot_center(self, reading_cm: float, heading: float) -> float:
        heading_to_wall_distance_cm = reading_cm + ROBOT_SENSOR_OFFSET_CM
        # Angle of incidence: heading deviation from nearest wall-perpendicular (0/90/180/270).
        angle_of_incidence = heading % 90
        if angle_of_incidence > 45:
            angle_of_incidence = 90 - angle_of_incidence
        return abs(heading_to_wall_distance_cm * math.cos(math.radians(angle_of_incidence)))

    def ultrasonic_reading_inside_robot_radius(self, reading_cm: float, heading: float) -> bool:
        distance = self.ultrasonic_to_robot_center(reading_cm, heading)
        robot_distance_to_wall_cm = distance - ROBOT_HALF_EXTENT_CM
        return robot_distance_to_wall_cm < WALL_CLEARANCE_CM
    
    def is_valid_point(self, x: float, y: float, extra_margin: float = 0.0) -> bool:
        x_lo = arena.ARENA_WEST_WALL_X_CM + WALL_CLEARANCE_CM + ROBOT_HALF_EXTENT_CM + extra_margin
        x_hi = arena.ARENA_EAST_WALL_X_CM - WALL_CLEARANCE_CM - ROBOT_HALF_EXTENT_CM - extra_margin
        y_lo = arena.ARENA_SOUTH_WALL_Y_CM + WALL_CLEARANCE_CM + ROBOT_HALF_EXTENT_CM + extra_margin
        y_hi = arena.ARENA_NORTH_WALL_Y_CM - WALL_CLEARANCE_CM - ROBOT_HALF_EXTENT_CM - extra_margin

        return  x > x_lo and x < x_hi and y > y_lo and y < y_hi

