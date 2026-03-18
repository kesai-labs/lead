"""AlpaSim metric recorder for producing route-level evaluation JSON files."""

import json
import logging
import os
import pathlib

import carla
import numpy as np
from agents.navigation.local_planner import RoadOption

LOG = logging.getLogger(__name__)


class AlpaSimMetricRecorder:
    """Collects per-step trajectory data and writes the AlpaSim JSON on finalise."""

    def __init__(self, route_original_name: str, output_path: str | os.PathLike):
        self.route_original_name = route_original_name
        self.output_path = pathlib.Path(output_path)
        self.agent_trajectory: list[dict] = []
        self.ground_truth_path: list[dict] = []

    # ------------------------------------------------------------------
    # Ground-truth path
    # ------------------------------------------------------------------

    def set_ground_truth_from_global_plan(
        self, global_plan_world_coord: list[tuple[carla.Transform, RoadOption]]
    ) -> None:
        """Build a 1 m-resolution ground-truth path from the leaderboard global plan.

        Args:
            global_plan_world_coord: As stored in ``AutonomousAgent._global_plan_world_coord``.
        """
        waypoints = np.array(
            [[t.location.x, t.location.y] for t, _ in global_plan_world_coord],
            dtype=float,
        )
        self.ground_truth_path = _resample_path_1m(waypoints)

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def record_step(self, x: float, y: float, heading: float, timestamp: int) -> None:
        """Append one trajectory waypoint.

        Args:
            x: CARLA world-frame x position (metres), from vehicle.get_transform().
            y: CARLA world-frame y position (metres), from vehicle.get_transform().
            heading: Heading in radians, converted from CARLA yaw (degrees, clockwise).
            timestamp: Integer step counter.
        """
        self.agent_trajectory.append(
            {
                "x": float(x),
                "y": float(y),
                "heading": float(heading),
                "timestamp": int(timestamp),
            }
        )

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def save(self, number_collisions: int) -> None:
        """Write the JSON file.

        Args:
            number_collisions: Total number of collision events on this route.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "route_original_name": self.route_original_name,
            "number_collisions": int(number_collisions),
            "agent_trajectory": self.agent_trajectory,
            "ground_truth_path": self.ground_truth_path,
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        LOG.info(
            "[AlpaSimMetricRecorder] Saved %d trajectory steps to %s",
            len(self.agent_trajectory),
            self.output_path,
        )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _resample_path_1m(waypoints: np.ndarray) -> list[dict]:
    """Resample a sequence of (x, y) waypoints to 1 m spacing.

    Args:
        waypoints: ``(N, 2)`` array of (x, y) positions.

    Returns:
        List of ``{"x": float, "y": float}`` dicts at ~1 m intervals.
    """
    if len(waypoints) < 2:
        return (
            [{"x": float(waypoints[0, 0]), "y": float(waypoints[0, 1])}]
            if len(waypoints) == 1
            else []
        )

    # Cumulative arc-length along the original path
    deltas = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(deltas)])
    total_length = cumlen[-1]

    if total_length == 0:
        return [{"x": float(waypoints[0, 0]), "y": float(waypoints[0, 1])}]

    # Sample at 1 m intervals
    sample_distances = np.arange(0.0, total_length, 1.0)
    xs = np.interp(sample_distances, cumlen, waypoints[:, 0])
    ys = np.interp(sample_distances, cumlen, waypoints[:, 1])

    return [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys, strict=True)]
