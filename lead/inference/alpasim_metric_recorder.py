"""AlpaSim metric recorder for producing route-level evaluation JSON files."""

import json
import logging
import os
import pathlib

import carla
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from beartype import beartype

LOG = logging.getLogger(__name__)


class AlpaSimMetricRecorder:
    """Collects per-step trajectory data and writes the AlpaSim JSON on finalise."""

    @beartype
    def __init__(
        self,
        route_original_name: str,
        output_path: str | os.PathLike,
        agent_name: str,
        team_name: str,
        model_name: str,
        vehicle: carla.Actor,
        global_plan_world_coord: list[tuple[carla.Transform, RoadOption]],
        carla_world: carla.World,
        decimal_places: int = 2,
    ) -> None:
        """Initialize the recorder and compute the dense ground-truth path.

        Args:
            route_original_name: Identifier for the route being evaluated.
            output_path: Destination for the JSON output file.
            agent_name: Name of the agent being evaluated.
            team_name: Team name for the evaluation entry.
            model_name: Model name for the evaluation entry.
            vehicle: The ego vehicle actor to track.
            global_plan_world_coord: Leaderboard global plan as (Transform, RoadOption) pairs.
            carla_world: CARLA world instance (used to obtain the map).
            decimal_places: Rounding precision for output coordinates.
        """
        self.route_original_name = route_original_name
        self.output_path = pathlib.Path(output_path)
        self.agent_name = agent_name
        self.team_name = team_name
        self.model_name = model_name
        self._vehicle = vehicle
        self._dp = decimal_places
        self.agent_trajectory: list[dict[str, list[float] | list[str]]] = []

        carla_map = carla_world.get_map()
        self.ground_truth_path = _dense_route_via_grp(
            global_plan_world_coord, carla_map, decimal_places=decimal_places
        )

    def record_step(self, infractions: list[str]) -> None:
        """Append one trajectory waypoint from the vehicle's current transform.

        Args:
            infractions: Infraction descriptions that occurred at this waypoint.
        """
        t = self._vehicle.get_transform()
        self.agent_trajectory.append(
            {
                "pose": [
                    round(t.location.x, self._dp),
                    round(t.location.y, self._dp),
                    round(np.deg2rad(t.rotation.yaw), self._dp),
                ],
                "infractions": infractions,
            }
        )

    def save(self, avg_score: float | None = None) -> None:
        """Write the JSON file.

        Args:
            avg_score: Route score in [0, 1]. Pass ``score_composed / 100`` from the
                leaderboard's statistics manager.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_name": self.agent_name,
            "team_name": self.team_name,
            "model_name": self.model_name,
            "avg_score": round(avg_score, self._dp) if avg_score is not None else None,
            "evaluation": [
                {
                    "route": self.route_original_name,
                    "trajectory": self.agent_trajectory,
                    "gt_path": self.ground_truth_path,
                }
            ],
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        LOG.info(
            "[AlpaSimMetricRecorder] Saved %d trajectory steps to %s",
            len(self.agent_trajectory),
            self.output_path,
        )


@beartype
def _dense_route_via_grp(
    global_plan_world_coord: list[tuple[carla.Transform, RoadOption]],
    carla_map: carla.Map,
    sampling_resolution: float = 1.0,
    decimal_places: int = 2,
) -> list[list[float]]:
    """Reconstruct the route at high density using CARLA's GlobalRoutePlanner.

    Args:
        global_plan_world_coord: Leaderboard global plan as (Transform, RoadOption) pairs.
        carla_map: CARLA map instance.
        sampling_resolution: Spacing between output waypoints in metres.
        decimal_places: Rounding precision for the output coordinates.

    Returns:
        List of [x, y] pairs at approximately sampling_resolution metre intervals.
    """
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    path: list[list[float]] = []
    for i in range(len(global_plan_world_coord) - 1):
        loc_a = global_plan_world_coord[i][0].location
        loc_b = global_plan_world_coord[i + 1][0].location
        for wp, _ in grp.trace_route(loc_a, loc_b):
            path.append(
                [
                    round(wp.transform.location.x, decimal_places),
                    round(wp.transform.location.y, decimal_places),
                ]
            )
    return path
