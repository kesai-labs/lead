"""
Expert agent with Py123D data logging.

This extends the Expert class to add Py123D Arrow format saving without modifying LEAD's core data processing or driving logic.
"""

import logging
import os
import typing
from collections import defaultdict
from pathlib import Path

import carla
import numpy as np
from beartype import beartype

try:
    from py123d.conversion.dataset_converter_config import DatasetConverterConfig
    from py123d.conversion.log_writer.abstract_log_writer import LiDARData
    from py123d.conversion.log_writer.arrow_log_writer import ArrowLogWriter, CameraData
    from py123d.conversion.map_writer.arrow_map_writer import ArrowMapWriter
    from py123d.conversion.registry.box_detection_label_registry import (
        DefaultBoxDetectionLabel,
    )
    from py123d.conversion.registry.lidar_index_registry import CARLALiDARIndex
    from py123d.conversion.utils.map_utils.opendrive.opendrive_map_conversion import (
        convert_xodr_map,
    )
    from py123d.datatypes.detections.box_detections import (
        BoxDetectionMetadata,
        BoxDetectionSE3,
        BoxDetectionWrapper,
    )
    from py123d.datatypes.detections.traffic_light_detections import (
        TrafficLightDetectionWrapper,
    )
    from py123d.datatypes.metadata import LogMetadata, MapMetadata
    from py123d.datatypes.sensors.lidar import LiDARMetadata, LiDARType
    from py123d.datatypes.sensors.pinhole_camera import (
        PinholeCameraMetadata,
        PinholeCameraType,
        PinholeIntrinsics,
    )
    from py123d.datatypes.time.time_point import TimePoint
    from py123d.datatypes.vehicle_state.ego_state import DynamicStateSE3, EgoStateSE3
    from py123d.datatypes.vehicle_state.vehicle_parameters import (
        get_carla_lincoln_mkz_2020_parameters,
    )
    from py123d.geometry import PoseSE3, Vector3D
    from py123d.script.utils.dataset_path_utils import get_dataset_paths
except Exception as e:
    print(
        f"Run 'pip install git+https://github.com/autonomousvision/py123d.git@dev_v0.0.9' to install Py123D. Import error: {e}"
    )
    raise e
from lead.common import constants
from lead.common.logging_config import setup_logging
from lead.expert import expert_py123d_utils
from lead.expert.expert import Expert

setup_logging()
LOG = logging.getLogger(__name__)


def get_entry_point() -> str:
    return "ExpertPy123D"


class ExpertPy123D(Expert):
    """Expert agent with Py123D data logging.

    This class extends Expert to add Py123D Arrow format saving
    without changing LEAD's core data processing or driving logic.
    """

    @beartype
    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ) -> None:
        """Setup agent with Py123D initialization.

        Args:
            path_to_conf_file: Path to configuration file.
            route_index: Optional route index identifier.
            traffic_manager: Optional CARLA traffic manager.
        """
        LOG.info("Starting setup...")
        LOG.info(f"Config file: {path_to_conf_file}")
        LOG.info(f"Route index: {route_index}")

        # Call parent setup - LEAD processes everything normally
        super().setup(path_to_conf_file, route_index, traffic_manager)

        LOG.info("Initializing Py123D writers...")

        # Get Py123D paths
        dataset_paths = get_dataset_paths()
        self._py123d_logs_root = Path(dataset_paths.py123d_logs_root)
        self._py123d_maps_root = Path(dataset_paths.py123d_maps_root)
        LOG.info(f"Py123D logs root: {self._py123d_logs_root}")
        LOG.info(f"Py123D maps root: {self._py123d_maps_root}")

        # Create directories
        self._py123d_logs_root.mkdir(parents=True, exist_ok=True)
        self._py123d_maps_root.mkdir(parents=True, exist_ok=True)
        LOG.info(f"Created logs directory: {self._py123d_logs_root.absolute()}")
        LOG.info(f"Created maps directory: {self._py123d_maps_root.absolute()}")

        # Initialize Py123D writers
        self._py123d_log_writer = ArrowLogWriter(logs_root=self._py123d_logs_root)
        self._py123d_map_writer = ArrowMapWriter(maps_root=self._py123d_maps_root)

        # Vehicle parameters (fixed for CARLA)
        self._vehicle_parameters = get_carla_lincoln_mkz_2020_parameters()

        # Dataset converter config
        self._dataset_converter_config = DatasetConverterConfig(
            force_log_conversion=True,
            force_map_conversion=False,
            include_map=True,
            include_ego=True,
            include_box_detections=True,
            include_traffic_lights=False,  # Not implemented yet
            include_pinhole_cameras=True,
            pinhole_camera_store_option="jpeg_binary",
            include_lidars=True,
            lidar_store_option="laz_binary",
            include_scenario_tags=False,
            include_route=False,
        )

        LOG.info("Py123D writers initialized")

    @beartype
    def _init(self, hd_map: carla.Map | None) -> None:
        """Initialize agent with Py123D map and log setup.

        Args:
            hd_map: Optional CARLA HD map.
        """
        # Call parent init - LEAD initializes everything
        super()._init(hd_map)

        LOG.info("Initializing Py123D map and log...")

        # Get location name
        self._location = self.carla_world.get_map().name.split("/")[-1].lower()
        self._log_name = f"{self.town}_Rep{self.rep}_{self.route_index}"

        LOG.info(f"Location: {self._location}")
        LOG.info(f"Log name: {self._log_name}")

        # Initialize map writer
        self._init_py123d_map()

        # Initialize log writer
        self._init_py123d_log()

        LOG.info("Py123D initialization complete")

    @beartype
    def _init_py123d_map(self) -> None:
        """Initialize Py123D map writer and convert OpenDRIVE map if needed."""
        LOG.info(f"Initializing map for location: {self._location}")
        LOG.info(f"Map output directory: {self._py123d_maps_root.absolute()}")

        map_metadata = MapMetadata(
            dataset="carla",
            split=None,
            log_name=None,
            location=self._location,
            map_has_z=True,
            map_is_local=False,
        )

        # Check if map needs conversion
        if self._py123d_map_writer.reset(self._dataset_converter_config, map_metadata):
            LOG.info(f"Converting map for {self._location}...")

            # Get CARLA root
            carla_root = Path(
                os.environ.get("CARLA_ROOT", os.getcwd() + "/3rd_party/CARLA_0915")
            )
            LOG.info(f"CARLA_ROOT: {carla_root.absolute()}")

            if not carla_root.exists():
                LOG.warning(
                    f"CARLA_ROOT not found: {carla_root.absolute()}. Map conversion skipped."
                )
            else:
                xodr_file = carla_root / constants.CARLA_MAP_PATHS.get(
                    self._location, ""
                )
                LOG.info(f"Looking for OpenDRIVE file: {xodr_file.absolute()}")

                if xodr_file.exists():
                    try:
                        LOG.info(
                            f"Starting map conversion from: {xodr_file.absolute()}"
                        )
                        convert_xodr_map(xodr_file, self._py123d_map_writer)
                        LOG.info(
                            f"Map conversion complete. Saved to: {self._py123d_maps_root.absolute()}"
                        )
                    except Exception as e:
                        LOG.error(f"Map conversion failed: {e}")
                else:
                    LOG.warning(f"OpenDRIVE file not found: {xodr_file.absolute()}")
        else:
            LOG.info(f"Map already converted for {self._location}")

        self._py123d_map_writer.close()
        LOG.info("Map writer closed")

    @beartype
    def _init_py123d_log(self) -> None:
        """Initialize Py123D log writer with camera and LiDAR metadata."""
        LOG.info(f"Initializing log writer for: {self._log_name}")
        LOG.info(f"Log output directory: {self._py123d_logs_root.absolute()}")

        # Get camera metadata
        LOG.info("Setting up camera metadata...")
        camera_metadata = self._get_py123d_camera_metadata()
        LOG.info(f"Camera metadata: {list(camera_metadata.keys())}")

        # Get LiDAR metadata
        LOG.info("Setting up LiDAR metadata...")
        lidar_metadata = self._get_py123d_lidar_metadata()
        LOG.info(f"LiDAR metadata: {list(lidar_metadata.keys())}")

        # Create log metadata
        log_metadata = LogMetadata(
            dataset="carla",
            split="train",
            log_name=self._log_name,
            location=self._location,
            timestep_seconds=self.config_expert.py123d_timestep_seconds,
            vehicle_parameters=self._vehicle_parameters,
            box_detection_label_class=DefaultBoxDetectionLabel,
            pinhole_camera_metadata=camera_metadata,
            lidar_metadata=lidar_metadata,
            map_metadata=MapMetadata(
                dataset="carla",
                split=None,
                log_name=None,
                location=self._location,
                map_has_z=True,
                map_is_local=False,
            ),
        )

        LOG.info(
            f"Log metadata created - log_name={self._log_name}, location={self._location}"
        )
        self._py123d_log_writer.reset(self._dataset_converter_config, log_metadata)
        LOG.info(
            f"Log writer initialized. Data will be saved to: {self._py123d_logs_root.absolute()}/{self._log_name}"
        )

    @beartype
    def _get_py123d_camera_metadata(
        self,
    ) -> dict[PinholeCameraType, PinholeCameraMetadata]:
        """Get camera metadata for Py123D from CARLA camera configuration.

        Returns:
            Dict mapping camera type to camera metadata.
        """
        # Use PCAM_F0 as the main camera (front center)
        camera_type = PinholeCameraType.PCAM_F0

        # Calculate intrinsics from CARLA camera parameters
        width = self.config_expert.camera_calibration[1]["width"]
        height = self.config_expert.camera_calibration[1]["height"]
        fov = self.config_expert.camera_calibration[1]["fov"]

        # https://github.com/carla-simulator/carla/issues/56
        focal_length = width / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = width / 2.0
        cy = height / 2.0

        intrinsics = PinholeIntrinsics(fx=focal_length, fy=focal_length, cx=cx, cy=cy)

        camera_metadata = PinholeCameraMetadata(
            camera_name=str(camera_type),
            camera_type=camera_type,
            intrinsics=intrinsics,
            distortion=None,
            width=width,
            height=height,
        )

        return {camera_type: camera_metadata}

    @beartype
    def _get_py123d_lidar_metadata(self) -> dict[LiDARType, LiDARMetadata]:
        """Get LiDAR metadata for Py123D from CARLA LiDAR configuration.

        Returns:
            Dict mapping LiDAR type to LiDAR metadata.
        """
        lidar_type = LiDARType.LIDAR_TOP

        # Get LiDAR extrinsic (relative to ego rear axle)
        lidar_pos = self.config_expert.lidar_pos_1
        lidar_rot = self.config_expert.lidar_rot_1

        quaternion = expert_py123d_utils.quaternion_from_carla_rotation(
            carla.Rotation(roll=lidar_rot[0], pitch=lidar_rot[1], yaw=lidar_rot[2])
        )

        lidar_extrinsic = PoseSE3(
            x=lidar_pos[0],
            y=-lidar_pos[1],  # Invert Y
            z=lidar_pos[2],
            qw=quaternion.qw,
            qx=quaternion.qx,
            qy=quaternion.qy,
            qz=quaternion.qz,
        )

        lidar_metadata = LiDARMetadata(
            lidar_name=str(lidar_type),
            lidar_type=lidar_type,
            lidar_index=CARLALiDARIndex,
            extrinsic=lidar_extrinsic,
        )

        return {lidar_type: lidar_metadata}

    @beartype
    def run_step(
        self,
        input_data: dict,
        timestamp: float,
        sensors: list[list[str | typing.Any]] | None,
    ) -> carla.VehicleControl:
        """Run step with Py123D data saving at 10 Hz.

        Args:
            input_data: Sensor data from CARLA.
            timestamp: Current simulation timestamp in seconds.
            sensors: Optional sensor configuration list.

        Returns:
            Vehicle control command.
        """
        # Call parent run_step - LEAD's driving logic runs unchanged
        control = super().run_step(input_data, timestamp, sensors)

        # Save to Py123D format at configured interval
        if (
            self.config_expert.datagen
            and self.step % self.config_expert.py123d_save_interval == 0
        ):
            py123d_data = self._convert_to_py123d(input_data, timestamp)
            self._py123d_log_writer.write(**py123d_data)
            if self.step % self.config_expert.py123d_log_interval == 0:
                LOG.info(
                    f"Saved Py123D data at step {self.step} (timestamp={timestamp:.2f}s)"
                )
        elif (
            self.step % self.config_expert.py123d_debug_log_interval == 0
            and self.config_expert.datagen
        ):
            LOG.debug(
                f"Step {self.step} - Py123D data saving ongoing to: {self._py123d_logs_root.absolute()}/{self._log_name}"
            )

        return control

    @beartype
    def _convert_to_py123d(self, input_data: dict, timestamp: float) -> dict:
        """Convert LEAD's processed data to Py123D format.

        Args:
            input_data: Sensor data from CARLA.
            timestamp: Current simulation timestamp in seconds.

        Returns:
            Dict with Py123D formatted data for log writer.
        """
        timestamp_pt = TimePoint.from_s(timestamp)

        # Extract ego state
        ego_state = self._extract_py123d_ego_state()

        # Convert LEAD's bounding boxes format to Py123D
        box_detections = self._extract_py123d_box_detections(input_data, timestamp_pt)
        LOG.debug(f"Extracted {len(box_detections.box_detections)} bounding boxes")

        # Convert camera data
        camera_data = self._extract_py123d_camera_data(input_data, timestamp_pt)
        LOG.debug(f"Extracted {len(camera_data)} camera frames")

        # Convert LiDAR data
        lidar_data = self._extract_py123d_lidar_data(timestamp_pt)
        if lidar_data:
            LOG.debug(
                f"Extracted LiDAR with {lidar_data[0].point_cloud.shape[0]} points"
            )

        # Traffic lights (empty for now)
        traffic_lights = TrafficLightDetectionWrapper(traffic_light_detections=[])

        return {
            "timestamp": timestamp_pt,
            "ego_state": ego_state,
            "box_detections": box_detections,
            "traffic_lights": traffic_lights,
            "pinhole_cameras": camera_data,
            "lidars": lidar_data,
        }

    @beartype
    def _extract_py123d_ego_state(self) -> EgoStateSE3:
        """Extract ego state from CARLA in Py123D format.

        Returns:
            EgoStateSE3 with position, velocity, and acceleration.
        """
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_acceleration = self.ego_vehicle.get_acceleration()
        ego_angular_velocity = self.ego_vehicle.get_angular_velocity()

        ego_center_se3 = expert_py123d_utils.carla_actor_to_se3(self.ego_vehicle)

        # Create ego dynamic state
        # Note: velocity and acceleration have Y-axis inverted for ISO 8855
        # Angular velocity does NOT have Y-axis inverted
        dynamic_state = DynamicStateSE3(
            velocity=Vector3D.from_list(
                [
                    ego_velocity.x,
                    -ego_velocity.y,
                    ego_velocity.z,
                ]
            ),
            acceleration=Vector3D.from_list(
                [
                    ego_acceleration.x,
                    -ego_acceleration.y,
                    ego_acceleration.z,
                ]
            ),
            angular_velocity=Vector3D.from_list(
                [
                    ego_angular_velocity.x,
                    ego_angular_velocity.y,
                    ego_angular_velocity.z,
                ]
            ),
        )

        ego_state = EgoStateSE3.from_center(
            center_se3=ego_center_se3,
            vehicle_parameters=self._vehicle_parameters,
            dynamic_state_se3=dynamic_state,
        )

        return ego_state

    @beartype
    def _extract_py123d_box_detections(
        self, input_data: dict, timestamp: TimePoint
    ) -> BoxDetectionWrapper:
        """Convert LEAD's bounding boxes format to Py123D.

        Args:
            input_data: Sensor data containing bounding boxes.
            timestamp: Current simulation timepoint.

        Returns:
            BoxDetectionWrapper with all detected objects.
        """
        box_detections: list[BoxDetectionSE3] = []

        # Vehicles
        for vehicle in self.vehicles_inside_bev:
            if vehicle.id == self.ego_vehicle.id:
                continue

            # Classify as bicycle or vehicle
            if vehicle.type_id in constants.BIKER_MESHES:
                label = DefaultBoxDetectionLabel.BICYCLE
            else:
                label = DefaultBoxDetectionLabel.VEHICLE

            box_detections.append(
                BoxDetectionSE3(
                    metadata=BoxDetectionMetadata(
                        label=label,
                        track_token=str(vehicle.id),
                        timepoint=timestamp,
                    ),
                    bounding_box_se3=expert_py123d_utils.get_actor_bounding_box_se3(
                        vehicle
                    ),
                    velocity_3d=expert_py123d_utils.carla_velocity_to_vector3d(
                        vehicle.get_velocity()
                    ),
                )
            )

        # Pedestrians
        for walker in self.walkers_inside_bev:
            box_detections.append(
                BoxDetectionSE3(
                    metadata=BoxDetectionMetadata(
                        label=DefaultBoxDetectionLabel.PERSON,
                        track_token=str(walker.id),
                        timepoint=timestamp,
                    ),
                    bounding_box_se3=expert_py123d_utils.get_actor_bounding_box_se3(
                        walker
                    ),
                    velocity_3d=expert_py123d_utils.carla_velocity_to_vector3d(
                        walker.get_velocity()
                    ),
                )
            )

        # Static bounding boxes
        for bb in input_data["bounding_boxes"]:
            if bb["class"] in ["static", "static_prop_car"]:
                overwrite_extent = bb["extent"]
                actor = self.id2actor_map.get(bb["id"])
                if (
                    bb["class"] == "static"
                    and "mesh_path" in bb
                    and "Car" in bb["mesh_path"]
                ):
                    box_detections.append(
                        BoxDetectionSE3(
                            metadata=BoxDetectionMetadata(
                                label=DefaultBoxDetectionLabel.VEHICLE,
                                track_token=str(bb["id"]),
                                timepoint=None,
                            ),
                            bounding_box_se3=expert_py123d_utils.get_bounding_box_se3(
                                bb
                            ),
                            velocity_3d=expert_py123d_utils.get_bounding_box_velocity(
                                bb
                            ),
                        )
                    )
                    LOG.info(f"Parking car detected: {bb['mesh_path']}")
                elif actor is None:  # static_prob_car that is not spawned as an actor
                    assert bb["class"] == "static_prop_car"
                    box_detections.append(
                        BoxDetectionSE3(
                            metadata=BoxDetectionMetadata(
                                label=DefaultBoxDetectionLabel.VEHICLE,
                                track_token=str(bb["id"]),
                                timepoint=None,
                            ),
                            bounding_box_se3=expert_py123d_utils.get_bounding_box_se3(
                                bb
                            ),
                            velocity_3d=expert_py123d_utils.get_bounding_box_velocity(
                                bb
                            ),
                        )
                    )
                else:
                    type_id_to_py123d_mapping = {
                        "static.prop.streetbarrier": DefaultBoxDetectionLabel.BARRIER,
                        "static.prop.constructioncone": DefaultBoxDetectionLabel.TRAFFIC_CONE,
                        "static.prop.trafficcone01": DefaultBoxDetectionLabel.TRAFFIC_CONE,
                        "static.prop.trafficcone02": DefaultBoxDetectionLabel.TRAFFIC_CONE,
                        "static.prop.trafficwarning": DefaultBoxDetectionLabel.TRAFFIC_SIGN,
                    }
                    if (
                        bb["class"] in ["car", "vehicle", "static_prop_car"]
                        or bb["type_id"] in type_id_to_py123d_mapping
                    ):
                        label = defaultdict(
                            lambda: DefaultBoxDetectionLabel.VEHICLE,
                            type_id_to_py123d_mapping,
                        )[bb["type_id"]]
                        LOG.info(f"{bb['type_id']} classified as {label}")
                        box_detections.append(
                            BoxDetectionSE3(
                                metadata=BoxDetectionMetadata(
                                    label=label,
                                    track_token=str(actor.id),
                                    timepoint=None,
                                ),
                                bounding_box_se3=expert_py123d_utils.get_actor_bounding_box_se3(
                                    actor, overwrite_extent=overwrite_extent
                                ),
                                velocity_3d=expert_py123d_utils.get_actor_velocity(
                                    actor
                                ),
                            )
                        )

        return BoxDetectionWrapper(box_detections=box_detections)

    @beartype
    def _extract_py123d_camera_data(
        self, input_data: dict, timestamp: TimePoint
    ) -> list[CameraData]:
        """Extract camera data from LEAD's input_data.

        Args:
            input_data: Sensor data containing RGB images.
            timestamp: Current simulation timepoint.

        Returns:
            List of CameraData with RGB images.
        """
        camera_data_list = []
        camera_type = PinholeCameraType.PCAM_F0

        # Get camera extrinsic with proper ISO 8855 conversion
        camera_pos = self.config_expert.camera_calibration[1]["pos"]
        camera_rot = self.config_expert.camera_calibration[1]["rot"]

        camera_extrinsic = expert_py123d_utils.get_camera_extrinsic_as_iso(
            camera_pos=camera_pos,
            camera_rot=camera_rot,
            vehicle_parameters=self._vehicle_parameters,
        )

        # Get RGB image from LEAD (use first camera)
        rgb_image = (
            input_data["rgb_1"][1][:, :, :3]
            if isinstance(input_data["rgb_1"], tuple)
            else input_data["rgb_1"][:, :, :3]
        )

        camera_data_list.append(
            CameraData(
                camera_name=str(camera_type),
                camera_type=camera_type,
                extrinsic=camera_extrinsic,
                timestamp=timestamp,
                numpy_image=rgb_image,
            )
        )

        return camera_data_list

    @beartype
    def _extract_py123d_lidar_data(self, timestamp: TimePoint) -> list[LiDARData]:
        """Extract LiDAR data from accumulated point cloud.

        Args:
            timestamp: Current simulation timepoint.

        Returns:
            List of LiDARData with point cloud.
        """
        lidar_type = LiDARType.LIDAR_TOP
        lidar_pc_py123d = self.accumulate_lidar().copy()

        lidar_pc_py123d[:, CARLALiDARIndex.Y] = -lidar_pc_py123d[:, CARLALiDARIndex.Y]
        lidar_pc_py123d[:, CARLALiDARIndex.X] += (
            self._vehicle_parameters.rear_axle_to_center_longitudinal
        )
        lidar_pc_py123d[:, CARLALiDARIndex.Z] += self.config_expert.lidar_pos_1[-1] / 2

        # Set intensity to 1.0
        lidar_pc_py123d[..., -1] = 1.0

        lidar_data = LiDARData(
            lidar_name=str(lidar_type),
            lidar_type=lidar_type,
            timestamp=timestamp,
            iteration=None,
            dataset_root=None,
            relative_path=None,
            point_cloud=lidar_pc_py123d,
        )

        return [lidar_data]

    @beartype
    def destroy(self, results: typing.Any = None) -> None:
        """Cleanup agent and close Py123D writers.

        Args:
            results: Optional results to pass to parent destroy.
        """
        LOG.info("Closing Py123D writers...")
        LOG.info(
            f"Final log location: {self._py123d_logs_root.absolute()}/{self._log_name}"
        )
        LOG.info(f"Final map location: {self._py123d_maps_root.absolute()}")
        self._py123d_log_writer.close()
        super().destroy(results)
        LOG.info("Cleanup complete - data saved to Py123D format")
