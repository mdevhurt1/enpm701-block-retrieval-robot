# Arena geometry. x east, y north, origin SW corner. Heading 0=north, CW positive.
import math

# Arena 7 ft x 7 ft
ARENA_WIDTH_CM  = 213.0
ARENA_HEIGHT_CM = 213.0

# Launch zone: 2 ft x 2 ft in SW corner.
LAUNCH_ZONE_SIZE_CM = 60.0

# Collection zone: 3 ft x 3 ft in NW corner.
COLLECTION_ZONE_SIZE_CM = 90.0

# Walls (axis-aligned bounds)
ARENA_SOUTH_WALL_Y_CM = 0.0
ARENA_WEST_WALL_X_CM  = 0.0
ARENA_NORTH_WALL_Y_CM = ARENA_HEIGHT_CM
ARENA_EAST_WALL_X_CM  = ARENA_WIDTH_CM

# Block size
BLOCK_HEIGHT_CM = 5.5
BLOCK_WIDTH_LENGTH_CM = 4.0


def apply_motion(x: float, y: float, distance_cm: float, heading_deg: float) -> tuple:
    rad = math.radians(heading_deg)
    return x + distance_cm * math.sin(rad), y + distance_cm * math.cos(rad)


def in_launch_zone(x: float, y: float) -> bool:
    return 0.0 <= x <= LAUNCH_ZONE_SIZE_CM and 0.0 <= y <= LAUNCH_ZONE_SIZE_CM


def in_collection_zone(x: float, y: float) -> bool:
    return (0.0 <= x <= COLLECTION_ZONE_SIZE_CM
            and (ARENA_HEIGHT_CM - COLLECTION_ZONE_SIZE_CM) <= y <= ARENA_HEIGHT_CM)


# Aim point at the centre of the NW collection zone.
COLLECTION_ZONE_CENTER = (
    COLLECTION_ZONE_SIZE_CM / 2,
    ARENA_HEIGHT_CM - (COLLECTION_ZONE_SIZE_CM / 2),
)

# Aim point at the centre of the SW launch zone.
LAUNCH_ZONE_CENTER = (
    LAUNCH_ZONE_SIZE_CM / 2,
    LAUNCH_ZONE_SIZE_CM / 2,
)
