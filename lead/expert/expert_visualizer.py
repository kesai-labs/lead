import logging

import carla
from beartype import beartype

from lead.expert.expert_data import ExpertData

LOG = logging.getLogger(__name__)


class ExpertVisualizer(ExpertData):
    @beartype
    def expert_setup(
        self, path_to_conf_file: str, route_index: str | None = None, traffic_manager: carla.TrafficManager | None = None
    ):
        super().expert_setup(path_to_conf_file, route_index, traffic_manager)

    @beartype
    def expert_init(self, hd_map: carla.Map | None):
        """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map: The map object of the CARLA world.
        """
        super().expert_init(hd_map)

    @beartype
    def visualize_ego_bb(self, ego_bb_global: carla.BoundingBox):
        ego_vehicle_transform = self.ego_vehicle.get_transform()
        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(self.ego_vehicle.bounding_box.location)
        ego_bb_global = carla.BoundingBox(center_ego_bb_global, self.ego_vehicle.bounding_box.extent)
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.config_expert.visualize_bounding_boxes:
            self.carla_world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=self.config_expert.ego_vehicle_bb_color,
                life_time=self.config_expert.draw_life_time,
            )

    @beartype
    def visualize_lead_and_trailing_vehicles(self):
        if self.config_expert.visualize_internal_data:
            vehicle_list = ...

            leading_vehicle_ids = self.privileged_route_planner.compute_leading_vehicles(vehicle_list, self.ego_vehicle.id)
            trailing_vehicle_ids = self.privileged_route_planner.compute_trailing_vehicles(vehicle_list, self.ego_vehicle.id)

            for vehicle in vehicle_list:
                if vehicle.id in leading_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Leading Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.leading_vehicle_color,
                    )
                elif vehicle.id in trailing_vehicle_ids:
                    self.carla_world.debug.draw_string(
                        vehicle.get_location(),
                        f"Trailing Vehicle: {vehicle.get_velocity().length():.2f} m/s",
                        life_time=self.config_expert.draw_life_time,
                        color=self.config_expert.trailing_vehicle_color,
                    )

    @beartype
    def visualize_forecasted_bounding_boxes(
        self,
        predicted_bounding_boxes: dict[int, list[carla.BoundingBox]],
    ):
        if self.config_expert.visualize_bounding_boxes:
            dangerous_adversarial_actors_ids, safe_adversarial_actors_ids, ignored_adversarial_actors_ids = (
                self.adversarial_actors_ids
            )
            for _actor_idx, actors_forecasted_bounding_boxes in predicted_bounding_boxes.items():
                for bb in actors_forecasted_bounding_boxes:
                    color = self.config_expert.other_vehicles_forecasted_bbs_color
                    if _actor_idx in dangerous_adversarial_actors_ids or _actor_idx in safe_adversarial_actors_ids:
                        color = self.config_expert.adversarial_color
                    self.carla_world.debug.draw_box(
                        box=bb,
                        rotation=bb.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=self.config_expert.draw_life_time,
                    )

                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if vehicle_id in self.leading_vehicle_ids and not self.near_lane_change:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.leading_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )
                    elif vehicle_id in self.trailing_vehicle_ids:
                        vehicle = self.carla_world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0)
                        self.carla_world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config_expert.trailing_vehicle_color,
                            life_time=self.config_expert.draw_life_time,
                        )

    @beartype
    def visualize_pedestrian_bounding_boxes(self, nearby_pedestrians_bbs: list[list[carla.BoundingBox]]):
        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.config_expert.visualize_bounding_boxes:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self.carla_world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=self.config_expert.pedestrian_forecasted_bbs_color,
                        life_time=self.config_expert.draw_life_time,
                    )

    @beartype
    def visualize_traffic_lights(self, traffic_light: carla.TrafficLight, wp: carla.Waypoint, bounding_box: carla.BoundingBox):
        if self.config_expert.visualize_traffic_lights_bounding_boxes:
            if traffic_light.state == carla.TrafficLightState.Red:
                color = self.config_expert.red_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Yellow:
                color = self.config_expert.yellow_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Green:
                color = self.config_expert.green_traffic_light_color
            elif traffic_light.state == carla.TrafficLightState.Off:
                color = self.config_expert.off_traffic_light_color
            else:  # unknown
                color = self.config_expert.unknown_traffic_light_color

            self.carla_world.debug.draw_box(
                box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=0.051
            )

            self.carla_world.debug.draw_point(
                wp.transform.location + carla.Location(z=traffic_light.trigger_volume.location.z),
                size=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
            )

            self.carla_world.debug.draw_box(
                box=traffic_light.bounding_box,
                rotation=traffic_light.bounding_box.rotation,
                thickness=0.1,
                color=color,
                life_time=0.051,
            )

    @beartype
    def visualize_stop_signs(self, bounding_box_stop_sign: carla.BoundingBox, affects_ego: bool):
        if self.config_expert.visualize_bounding_boxes:
            color = carla.Color(0, 1, 0) if affects_ego else carla.Color(1, 0, 0)
            self.carla_world.debug.draw_box(
                box=bounding_box_stop_sign,
                rotation=bounding_box_stop_sign.rotation,
                thickness=0.1,
                color=color,
                life_time=(1.0 / self.config_expert.carla_fps) + 1e-6,
            )
