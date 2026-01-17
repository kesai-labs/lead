import json
import logging
import os
import shutil
from collections import deque
from copy import deepcopy

import carla
import cv2
import jaxtyping as jt
import matplotlib
import numpy as np
import numpy.typing as npt
import torch
from beartype import beartype
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.logging_config import setup_logging
from lead.common.route_planner import RoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.common.visualizer import Visualizer
from lead.data_loader import carla_dataset_utils, training_cache
from lead.data_loader.carla_dataset_utils import rasterize_lidar
from lead.expert import expert_utils
from lead.inference.closed_loop_inference import (
    ClosedLoopInference,
    ClosedLoopPrediction,
)
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.video_recorder import VideoRecorder
from lead.training.config_training import TrainingConfig

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():  # dead: disable
    return "SensorAgent"


class SensorAgent(BaseAgent, autonomous_agent.AutonomousAgent):
    @beartype
    def setup(self, path_to_conf_file: str, _=None, __=None):
        # Set up test time training default parameters
        self.config_closed_loop = ClosedLoopConfig()
        super().setup(sensor_agent=True)
        self.config_path = path_to_conf_file
        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")

        # Load the config saved during training
        if self.config_closed_loop.is_bench2drive:
            path_to_conf_file = path_to_conf_file.split("+")[0]
        with open(os.path.join(path_to_conf_file, "config.json"), encoding="utf-8") as f:
            json_config = f.read()
            json_config = json.loads(json_config)

        # Generate new config for the case that it has new variables.
        self.training_config = TrainingConfig(json_config)

        # Store training config in base class for Kalman filter decision
        # This is accessed by BaseAgent._use_kalman_filter()

        # Load model files
        self.closed_loop_inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            model_path=path_to_conf_file,
            device=self.device,
            prefix="model",
        )

        # Post-processing heuristics
        self.bb_buffer = deque(maxlen=1)
        self.force_move_post_processor = ForceMovePostProcessor(
            config=self.training_config, config_test_time=self.config_closed_loop, lidar_queue=self.lidar_pc_queue
        )
        self.metric_info = {}
        self.meters_travelled = 0.0

        # Infraction tracking
        self.infractions_log = []  # List of {"step": int, "infraction": str}
        self.tracked_infraction_ids = set()  # Track which infractions we've already logged
        self.scenario = None  # Will be set by set_scenario() method

        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression.")

    def set_scenario(self, scenario):
        """Set the scenario reference to track infractions.

        This should be called by the leaderboard after loading the scenario.
        """
        self.scenario = scenario
        LOG.info("[SensorAgent] Scenario reference set for infraction tracking")

    def _init(self):
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()

        # Set up video recorder
        self.video_recorder = VideoRecorder(
            config_closed_loop=self.config_closed_loop,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            training_config=self.training_config,
        )

        self.set_weather()
        self.initialized = True

    def set_weather(self):
        weather_name = None

        if self.config_closed_loop.random_weather:
            weathers = self.config_expert.weather_settings.keys()
            weather_name = np.random.choice(list(weathers))

        if self.config_closed_loop.custom_weather is not None:
            weather_name = self.config_closed_loop.custom_weather

        if weather_name is not None:
            weather = carla.WeatherParameters(**self.config_expert.weather_settings[weather_name])
            self._world.set_weather(weather)
            LOG.info(f"Set weather to: {weather_name}")
            # night mode
            vehicles = self._world.get_actors().filter("*vehicle*")
            if expert_utils.get_night_mode(weather):
                for vehicle in vehicles:
                    vehicle.set_light_state(
                        carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
                    )
            else:
                for vehicle in vehicles:
                    vehicle.set_light_state(carla.VehicleLightState.NONE)

    @beartype
    def sensors(self) -> list[dict]:
        return av_sensor_setup(
            config=self.training_config,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    def check_infractions(self) -> None:
        """Check for infractions that occurred in the current step and log them.

        Handles two types of infractions:
        1. Discrete events (e.g., collisions): Each event is logged once
        2. Continuous infractions (e.g., OutsideRouteLanesTest): Only logged when first detected

        This matches the behavior of scenario_runner where continuous infractions create
        a single event that gets updated, while discrete infractions create new events.
        """
        if self.scenario is None:
            return

        try:
            criteria = self.scenario.get_criteria()

            for criterion in criteria:
                if hasattr(criterion, "events") and criterion.events:
                    # Track which criterion is currently active
                    # For continuous infractions (like OutsideRouteLanesTest), we only log once
                    # until the infraction is cleared
                    criterion_key = criterion.name

                    for event in criterion.events:
                        # For discrete events (collisions), use frame as unique identifier
                        # For continuous events (route lanes), use only criterion name
                        # Continuous infractions typically have only 1 event that gets updated
                        is_continuous = len(criterion.events) == 1

                        if is_continuous:
                            # Continuous infraction: only log if not currently tracked
                            event_id = criterion_key
                        else:
                            # Discrete infraction: log each unique frame
                            event_id = (criterion_key, event.get_frame())

                        # Only log if we haven't seen this infraction before
                        if event_id not in self.tracked_infraction_ids:
                            self.tracked_infraction_ids.add(event_id)

                            # Map criterion name to readable infraction type
                            infraction_info = {
                                "step": self.step,
                                "infraction": criterion.name,
                                "frame": event.get_frame(),
                                "message": event.get_message() if hasattr(event, "get_message") else "",
                                "event_type": str(event.get_type()) if hasattr(event, "get_type") else "",
                                "meters_travelled": round(self.meters_travelled, 2),
                            }

                            self.infractions_log.append(infraction_info)
                            LOG.info(f"[SensorAgent] Infraction detected at step {self.step}: {criterion.name}")

                    # If no events remain for this criterion, remove it from tracking
                    # This allows continuous infractions to be logged again if they reoccur
                    if not criterion.events and criterion_key in self.tracked_infraction_ids:
                        self.tracked_infraction_ids.discard(criterion_key)

            # Save infractions log to JSON
            if self.config_closed_loop.save_path is not None and hasattr(self, "infractions_log"):
                infractions_path = self.config_closed_loop.save_path / "infractions.json"
                infractions_data = {
                    "infractions": self.infractions_log,
                    "video_fps": self.config_closed_loop.video_fps,
                }
                with open(infractions_path, "w") as f:
                    json.dump(infractions_data, f, indent=4)
                LOG.info(f"[SensorAgent] Saved {len(self.infractions_log)} infractions to {infractions_path}")

        except Exception as e:
            LOG.warning(f"[SensorAgent] Error checking infractions: {e}")

    @beartype
    def set_target_points(self, input_data: dict, pop_distance: float):
        """Defines local planning signals based on the input data.

        Args:
            input_data: The input data containing sensor information and state. Will be fed into model.
            pop_distance: Distance threshold to pop waypoints from the route planner.
        """
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]

        @beartype
        def transform(point: list[float]) -> jt.Float[npt.NDArray, " 2"]:
            # Use filtered or noisy position based on training config
            ego_position = (
                self.filtered_state[:2] if self.training_config.use_kalman_filter_for_gps else input_data["noisy_state"][:2]
            )
            return common_utils.inverse_conversion_2d(np.array(point), np.array(ego_position), self.compass)

        previous_target_points = [tp.tolist() for tp in planner.previous_target_points]
        next_target_points = [tp[0].tolist() for tp in planner.route]
        next_commands = [int(planner.route[i][1]) for i in range(len(planner.route))]

        # Merge duplicate consecutive target points
        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_target_points, next_commands, strict=False):
            if len(next_target_points) == 2 or not filtered_tp_list or not np.allclose(pt[:2], filtered_tp_list[-1][:2]):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)
        next_target_points = filtered_tp_list
        next_commands = filtered_command_list

        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            assert len(next_target_points) == 2
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            if len(previous_target_points) > 0:
                input_data["target_point_previous"] = transform(previous_target_points[-1][:2])
            else:
                input_data["target_point_previous"] = transform(next_target_points[0][:2])

        input_data["command"] = carla_dataset_utils.command_to_one_hot(next_commands[0])
        input_data["next_command"] = carla_dataset_utils.command_to_one_hot(next_commands[1])

    @beartype
    @torch.inference_mode()
    def tick(self, input_data: dict) -> dict:
        """Pre-processes sensor data"""
        input_data = super().tick(input_data, use_kalman_filter=self.training_config.use_kalman_filter_for_gps)

        # Simulate JPEG compression to avoid train-test mismatch
        rgb = input_data["rgb"]
        input_data["original_rgb"] = rgb.copy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        _, rgb = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.config_closed_loop.jpeg_quality])
        rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
        rgb = np.transpose(rgb, (2, 0, 1))
        input_data["rgb"] = rgb

        # Horizontal FOV reduction: crop left and right, then resize back
        if self.training_config.horizontal_fov_reduction > 0:
            crop_pixels = self.training_config.horizontal_fov_reduction
            # rgb: (C, H, W)
            if input_data["rgb"] is not None:
                _, h, w = input_data["rgb"].shape
                input_data["rgb"] = input_data["rgb"][:, :, crop_pixels:-crop_pixels]
                input_data["rgb"] = np.transpose(input_data["rgb"], (1, 2, 0))  # -> (H, W_crop, C)
                input_data["rgb"] = cv2.resize(input_data["rgb"], (w, h), interpolation=cv2.INTER_LINEAR)
                input_data["rgb"] = np.transpose(input_data["rgb"], (2, 0, 1))  # -> (C, H, W)
            # original_rgb: (H, W, C)
            if input_data["original_rgb"] is not None:
                h, w = input_data["original_rgb"].shape[:2]
                input_data["original_rgb"] = input_data["original_rgb"][:, crop_pixels:-crop_pixels, :]
                input_data["original_rgb"] = cv2.resize(input_data["original_rgb"], (w, h), interpolation=cv2.INTER_LINEAR)

        # Cut cameras down to only used cameras
        for modality in ["rgb", "original_rgb"]:
            if self.training_config.num_used_cameras != self.training_config.num_available_cameras:
                n = self.training_config.num_available_cameras
                w = input_data[modality].shape[2] // n

                rgb_slices = []
                for i, use in enumerate(self.training_config.used_cameras):
                    if use:
                        s, e = i * w, (i + 1) * w
                        rgb_slices.append(input_data[modality][:, :, s:e])

                input_data[modality] = np.concatenate(rgb_slices, axis=2)

        # Plan next target point and command.
        self.set_target_points(input_data, pop_distance=self.config_closed_loop.route_planner_min_distance)
        if self.config_closed_loop.sensor_agent_pop_distance_adaptive:
            dense_points = (
                np.linalg.norm(input_data["target_point"] - input_data["target_point_next"]) < 10.0
                and min(np.linalg.norm(input_data["target_point_previous"]), np.linalg.norm(input_data["target_point"])) < 10.0
            )
            dense_points = dense_points or (
                np.linalg.norm(input_data["target_point_previous"] - input_data["target_point"]) < 10.0
                and min(np.linalg.norm(input_data["target_point_previous"]), np.linalg.norm(input_data["target_point"])) < 10.0
            )
            if dense_points:
                self.set_target_points(input_data, pop_distance=4.0)

        # Ignore the next target point if it's too far away
        if (
            self.config_closed_loop.sensor_agent_skip_distant_target_point
            and np.linalg.norm(input_data["target_point_next"])
            > self.config_closed_loop.sensor_agent_skip_distant_target_point_threshold
        ):
            # Skip the next target point if it's too far away
            input_data["target_point_next"] = input_data["target_point"]

        # Lidar input
        lidar = self.accumulate_lidar()
        # Use only part of the lidar history we trained on
        lidar = lidar[lidar[:, -1] < self.training_config.training_used_lidar_steps]

        # At inference time, simulate laspy quantization to avoid train-test mismatch
        lidar[:, 0] = np.round(lidar[:, 0] / self.config_expert.point_precision_x) * self.config_expert.point_precision_x
        lidar[:, 1] = np.round(lidar[:, 1] / self.config_expert.point_precision_y) * self.config_expert.point_precision_y
        lidar[:, 2] = np.round(lidar[:, 2] / self.config_expert.point_precision_z) * self.config_expert.point_precision_z

        # Convert to pseudo image
        input_data["rasterized_lidar"] = rasterize_lidar(config=self.training_config, lidar=lidar[:, :3])[..., None]

        # Simulate training time compression to avoid train-test mismatch
        input_data["rasterized_lidar"] = training_cache.compress_float_image(
            input_data["rasterized_lidar"], self.training_config
        )
        input_data["rasterized_lidar"] = training_cache.decompress_float_image(input_data["rasterized_lidar"]).squeeze()[
            None, None
        ]

        # Radar input preprocessing
        if self.training_config.use_radars:
            # Preprocess radar input using the same function as during training
            input_data["radar"] = np.concatenate(
                carla_dataset_utils.preprocess_radar_input(self.training_config, input_data), axis=0
            )

        return input_data

    @beartype
    @torch.inference_mode()
    def run_step(self, input_data: dict, _, __=None) -> carla.VehicleControl:
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            input_data = self.tick(input_data)
            return self.control

        # Update video recorder step and demo cameras
        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        input_data = self.tick(input_data)

        # Transform the data into torch tensor comforting with data loader's format.
        input_data_tensors = {
            "rgb": torch.Tensor(input_data["rgb"]).to(self.device, dtype=torch.float32)[None],
            "rasterized_lidar": torch.Tensor(input_data["rasterized_lidar"]).to(self.device, dtype=torch.float32),
            "target_point_previous": torch.Tensor(input_data["target_point_previous"])
            .to(self.device, dtype=torch.float32)
            .view(1, 2),
            "target_point": torch.Tensor(input_data["target_point"]).to(self.device, dtype=torch.float32).view(1, 2),
            "target_point_next": (torch.Tensor(input_data["target_point_next"]).to(self.device, dtype=torch.float32)).view(
                1, 2
            ),
            "speed": torch.Tensor([input_data["speed"]]).to(self.device, dtype=torch.float32).view(1),
            "command": torch.Tensor(input_data["command"]).to(self.device, dtype=torch.float32).view(1, 6),
            "town": np.array([self._world.get_map().name]),
        }

        # Add radar data if available
        if self.training_config.use_radars and "radar" in input_data:
            input_data_tensors["radar"] = torch.Tensor(input_data["radar"]).to(self.device, dtype=torch.float32)[None]

        # Save input log if need
        if self.config_closed_loop.save_path is not None and self.config_closed_loop.produce_input_log:
            torch.save(
                {k: v.to(torch.device("cpu")) if isinstance(v, torch.Tensor) else v for k, v in input_data_tensors.items()},
                os.path.join(self.config_closed_loop.input_log_path, str(self.step).zfill(5)) + ".pth",
            )

        # Forward pass
        closed_loop_prediction: ClosedLoopPrediction = self.closed_loop_inference.forward(data=input_data_tensors)
        # Update bounding boxes
        if (
            closed_loop_prediction.pred_bounding_box_vehicle_system is not None
            and len(closed_loop_prediction.pred_bounding_box_vehicle_system) > 0
        ):
            self.bb_buffer.append(closed_loop_prediction.pred_bounding_box_vehicle_system)

        # Post-processing heuristic
        closed_loop_prediction.throttle, closed_loop_prediction.brake = self.force_move_post_processor.adjust(
            input_data["speed"].item(), closed_loop_prediction.throttle, closed_loop_prediction.brake
        )
        self.meters_travelled += input_data["speed"].item() * self.config_closed_loop.carla_frame_rate
        input_data["meters_travelled"] = self.meters_travelled

        self.control = carla.VehicleControl(
            steer=float(closed_loop_prediction.steer),
            throttle=float(closed_loop_prediction.throttle),
            brake=float(closed_loop_prediction.brake),
        )

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.training_config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # Check for infractions at this step
        self.check_infractions()

        # Visualization of prediction for debugging and video recording
        input_data_tensors.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]),
                "stuck_detector": torch.Tensor([self.force_move_post_processor.stuck_detector]),
                "force_move": torch.Tensor([self.force_move_post_processor.force_move]),
                "route_curvature": torch.Tensor(
                    [common_utils.waypoints_curvature(closed_loop_prediction.pred_route.squeeze())]
                ),
                "meters_travelled": torch.Tensor([self.meters_travelled]),
            }
        )

        # Save input images as PNG and video
        if self.config_closed_loop.save_path is not None and self.step % self.config_closed_loop.produce_frame_frequency == 0:
            # Get the RGB image for visualization (before JPEG compression)
            input_image = input_data["original_rgb"].copy()

            # Save input image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_input_image(input_image)
                self.video_recorder.save_input_video_frame(input_image)

        # Save demo images
        if self.config_closed_loop.save_path is not None and self.step % self.config_closed_loop.produce_frame_frequency == 0:
            # Get predicted route and waypoints (if available)
            pred_waypoints = (
                closed_loop_prediction.pred_future_waypoints[0]
                if closed_loop_prediction.pred_future_waypoints is not None
                else None
            )

            # Prepare target points dictionary for BEV visualization
            target_points = {
                "previous": input_data.get("target_point_previous"),
                "current": input_data.get("target_point"),
                "next": input_data.get("target_point_next"),
            }

            # Save demo cameras with visualization using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_demo_cameras(pred_waypoints, target_points)
                # Save grid (demo + input stacked vertically) with planning visualization
                self.video_recorder.save_grid_image_and_video(
                    pred_waypoints=pred_waypoints,
                    target_points=target_points,
                )

        # Save abstract debug images
        if (
            self.config_closed_loop.save_path is not None
            and (self.config_closed_loop.produce_debug_video or self.config_closed_loop.produce_debug_image)
            and self.step % self.config_closed_loop.produce_frame_frequency == 0
        ):
            # Produce image
            image = Visualizer(
                config=self.training_config,
                data=input_data_tensors,
                prediction=closed_loop_prediction,
                config_test_time=self.config_closed_loop,
                test_time=True,
            ).visualize_inference_prediction()
            image = np.array(image).astype(np.uint8)

            # Save debug image and video using VideoRecorder
            if hasattr(self, "video_recorder"):
                self.video_recorder.save_debug_video_frame(image)
                self.video_recorder.save_debug_image(image)

        # Save metric info if in Bench2Drive mode
        if self.config_closed_loop.is_bench2drive and hasattr(self, "get_metric_info"):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(f"{self.config_closed_loop.save_path}/metric_info.json", "w") as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, _=None):
        # Clean up video recorder
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup_and_compress()


class ForceMovePostProcessor:
    """Forces the agent to move after a certain time of being stuck."""

    @beartype
    def __init__(self, config: TrainingConfig, config_test_time: ClosedLoopConfig, lidar_queue: deque):
        self.config = config
        self.config_test_time = config_test_time
        self.stuck_detector = 0
        self.force_move = 0
        self.lidar_buffer = lidar_queue

    @beartype
    def adjust(self, ego_speed: float, current_throttle: float, current_brake: float) -> tuple[float, float]:
        if ego_speed < 0.1:  # 0.1 is just an arbitrary low number to threshold when the car is stopped
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # If last red light was encountered a long time ago, we can assume it was cleared
        stuck_threshold = self.config_test_time.sensor_agent_stuck_threshold

        if self.stuck_detector > stuck_threshold:
            self.force_move = self.config_test_time.sensor_agent_stuck_move_duration

        if self.force_move > 0:
            emergency_stop = False
            # safety check
            safety_box = deepcopy(self.lidar_buffer[-1])

            # z-axis
            safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
            if len(safety_box) > 0:  # Checks if the List is empty
                emergency_stop = True
                LOG.info("Creeping overriden by safety box.")
            if not emergency_stop:
                LOG.info("Detected agent being stuck.")
                current_throttle = max(self.config_test_time.sensor_agent_stuck_throttle, current_throttle)
                current_brake = 0.0
                self.force_move -= 1
            else:
                LOG.info("Forced moving stopped by safety box.")
                current_throttle = 0.0
                current_brake = 1.0
                self.force_move = self.config_test_time.sensor_agent_stuck_move_duration
        return current_throttle, current_brake


if __name__ == "__main__":
    sensor_agent = SensorAgent()
