import math

# Arena / zone dimensions

ARENA_WIDTH_CM          = 213.0
ARENA_HEIGHT_CM         = 213.0
LAUNCH_ZONE_SIZE_CM     =  61.0
COLLECTION_ZONE_SIZE_CM =  61.0

GRID_RESOLUTION_CM      =   5
GRID_SIZE               =  43

QR_LANDMARKS = {
    "x=107,y=107": (107.0, 107.0),
    "x=180,y=30":  (180.0,  30.0),
    "x=30,y=180":  ( 30.0, 180.0),
}

ROBOT_SENSOR_OFFSET_CM  =  15.0
ARENA_SOUTH_WALL_Y_CM   =   0.0
ARENA_WEST_WALL_X_CM    =   0.0
ARENA_NORTH_WALL_Y_CM   = ARENA_HEIGHT_CM
ARENA_EAST_WALL_X_CM    = ARENA_WIDTH_CM


def apply_motion(x: float, y: float, distance_cm: float, heading_deg: float) -> tuple:
    """Advance (x, y) by distance_cm along heading_deg (0 = north, CW positive)."""
    rad = math.radians(heading_deg)
    return x + distance_cm * math.sin(rad), y + distance_cm * math.cos(rad)
