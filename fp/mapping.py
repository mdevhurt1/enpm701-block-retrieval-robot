# Final Project - Path Tracking and Visualization
# PathTracker records (x, y) waypoints during a run and renders a
# color-coded multi-segment path overlaid on the arena layout.
#
# Requirements: matplotlib, colorsys
import colorsys
import logging
import random

logger = logging.getLogger(__name__)


class PathTracker:
    def __init__(self) -> None:
        self._segments: list[tuple[tuple[float, float, float], list[tuple[float, float]]]] = []

    def start_segment(self) -> None:
        hue = random.random()
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 0.85)
        self._segments.append(((r, g, b), []))

    def record(self, x: float, y: float) -> None:
        if not self._segments:
            return
        self._segments[-1][1].append((x, y))

    def save_plot(self, filepath: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        from fp import arena

        fig, ax = plt.subplots(figsize=(8, 8))

        # Arena background
        ax.add_patch(patches.Rectangle(
            (0, 0), arena.ARENA_WIDTH_CM, arena.ARENA_HEIGHT_CM,
            linewidth=2, edgecolor="black", facecolor="#f5f5f5",
        ))

        # Collection zone (NW corner)
        cz_y = arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM
        ax.add_patch(patches.Rectangle(
            (0, cz_y), arena.COLLECTION_ZONE_SIZE_CM, arena.COLLECTION_ZONE_SIZE_CM,
            linewidth=1, edgecolor="steelblue", facecolor="#d6eaf8",
        ))
        ax.text(
            arena.COLLECTION_ZONE_SIZE_CM / 2,
            cz_y + arena.COLLECTION_ZONE_SIZE_CM / 2,
            "Collection", ha="center", va="center", fontsize=8, color="steelblue",
        )

        # Launch zone (SW corner)
        ax.add_patch(patches.Rectangle(
            (0, 0), arena.LAUNCH_ZONE_SIZE_CM, arena.LAUNCH_ZONE_SIZE_CM,
            linewidth=1, edgecolor="green", facecolor="#d5f5e3",
        ))
        ax.text(
            arena.LAUNCH_ZONE_SIZE_CM / 2,
            arena.LAUNCH_ZONE_SIZE_CM / 2,
            "Launch", ha="center", va="center", fontsize=8, color="green",
        )

        # Path segments
        for color, points in self._segments:
            if len(points) < 2:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color=color, linewidth=1.5)

        ax.set_xlim(-5, arena.ARENA_WIDTH_CM + 5)
        ax.set_ylim(-5, arena.ARENA_HEIGHT_CM + 5)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title("Robot Path")
        ax.set_aspect("equal")

        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[MAPPING] path plot saved to {filepath} ({len(self._segments)} segments)")
