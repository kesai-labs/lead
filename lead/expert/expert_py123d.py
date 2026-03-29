"""
Expert agent with Py123D data logging.

This extends the Expert class to add Py123D Arrow format saving without modifying LEAD's core data processing or driving logic.
"""

import json
import logging
import os
import typing
from collections import defaultdict
from pathlib import Path

import carla
import numpy as np
from beartype import beartype
from py123d.api.map.arrow.arrow_map_writer import ArrowMapWriter
from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.datatypes.detections.box_detection_label import DefaultBoxDetectionLabel
from py123d.datatypes.detections.box_detections import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
)
from py123d.datatypes.detections.box_detections_metadata import (
    BoxDetectionsSE3Metadata,
)
from py123d.datatypes.detections.traffic_light_detections import (
    TrafficLightDetection,
    TrafficLightDetections,
)
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.sensors.base_camera import Camera, CameraID
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import (
    PinholeCameraMetadata,
    PinholeIntrinsics,
)
from py123d.datatypes.time.timestamp import Timestamp
from py123d.datatypes.vehicle_state.dynamic_state import DynamicStateSE3
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.vehicle_state.ego_state_metadata import (
    get_carla_lincoln_mkz_2020_metadata,
)
from py123d.geometry import PoseSE3, Vector3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.parser.base_dataset_parser import ModalitiesSync
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects
from py123d.script.utils.dataset_path_utils import get_dataset_paths

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

        # Log writer config
        self._log_writer_config = LogWriterConfig(
            force_log_conversion=True,
            camera_store_option="jpeg_binary",
            lidar_store_option="binary",
            lidar_codec="laz",
        )

        # Initialize Py123D writers
        self._py123d_sensors_root = Path(dataset_paths.py123d_sensors_root)
        self._py123d_sensors_root.mkdir(parents=True, exist_ok=True)
        self._py123d_log_writer = ArrowLogWriter(
            log_writer_config=self._log_writer_config,
            logs_root=self._py123d_logs_root,
            sensors_root=self._py123d_sensors_root,
        )
        self._py123d_map_writer = ArrowMapWriter(
            force_map_conversion=False,
            maps_root=self._py123d_maps_root,
            logs_root=self._py123d_logs_root,
        )

        # Vehicle metadata (fixed for CARLA)
        self._ego_metadata = get_carla_lincoln_mkz_2020_metadata()

        # Box detections metadata
        self._box_detections_metadata = BoxDetectionsSE3Metadata(
            box_detection_label_class=DefaultBoxDetectionLabel,
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
            location=self._location,
            map_has_z=True,
            map_is_per_log=False,
        )

        # Check if map needs conversion
        if self._py123d_map_writer.reset(map_metadata):
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
                        for map_object in iter_xodr_map_objects(xodr_file):
                            self._py123d_map_writer.write_map_object(map_object)
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
        """Initialize Py123D log writer."""
        LOG.info(f"Initializing log writer for: {self._log_name}")
        LOG.info(f"Log output directory: {self._py123d_logs_root.absolute()}")

        # Store camera/lidar metadata for use in extraction methods
        self._camera_metadata = self._get_py123d_camera_metadata()
        self._lidar_metadata = self._get_py123d_lidar_metadata()

        # Create log metadata (sensor metadata is now per-modality, not in LogMetadata)
        log_metadata = LogMetadata(
            dataset=self.config_expert.py123d_dataset,
            split=self.config_expert.py123d_split,
            log_name=self._log_name,
            location=self._location,
            map_metadata=MapMetadata(
                dataset=self.config_expert.py123d_dataset,
                location=self._location,
                map_has_z=True,
                map_is_per_log=False,
            ),
        )

        LOG.info(
            f"Log metadata created - log_name={self._log_name}, location={self._location}"
        )
        self._py123d_log_writer.reset(log_metadata)
        LOG.info(
            f"Log writer initialized. Data will be saved to: {self._py123d_logs_root.absolute()}/{self._log_name}"
        )

    @beartype
    def _get_py123d_camera_metadata(self) -> PinholeCameraMetadata:
        """Get camera metadata for Py123D from CARLA camera configuration.

        Returns:
            PinholeCameraMetadata for the front camera.
        """
        camera_id = CameraID.PCAM_F0

        # Calculate intrinsics from CARLA camera parameters
        width = self.config_expert.camera_calibration[1]["width"]
        height = self.config_expert.camera_calibration[1]["height"]
        fov = self.config_expert.camera_calibration[1]["fov"]

        # https://github.com/carla-simulator/carla/issues/56
        focal_length = width / (2.0 * np.tan(fov * np.pi / 360.0))
        cx = width / 2.0
        cy = height / 2.0

        intrinsics = PinholeIntrinsics(fx=focal_length, fy=focal_length, cx=cx, cy=cy)

        # Get camera extrinsic relative to IMU (rear axle)
        camera_pos = self.config_expert.camera_calibration[1]["pos"]
        camera_rot = self.config_expert.camera_calibration[1]["rot"]
        camera_to_imu_se3 = expert_py123d_utils.get_camera_extrinsic_as_iso(
            camera_pos=camera_pos,
            camera_rot=camera_rot,
            ego_metadata=self._ego_metadata,
        )

        return PinholeCameraMetadata(
            camera_name=str(camera_id),
            camera_id=camera_id,
            intrinsics=intrinsics,
            distortion=None,
            width=width,
            height=height,
            camera_to_imu_se3=camera_to_imu_se3,
        )

    @beartype
    def _get_py123d_lidar_metadata(self) -> LidarMetadata:
        """Get LiDAR metadata for Py123D from CARLA LiDAR configuration.

        Returns:
            LidarMetadata for the top LiDAR.
        """
        lidar_id = LidarID.LIDAR_TOP

        # Get LiDAR extrinsic (relative to IMU / rear axle)
        lidar_pos = self.config_expert.lidar_pos_1
        lidar_rot = self.config_expert.lidar_rot_1

        quaternion = expert_py123d_utils.quaternion_from_carla_rotation(
            carla.Rotation(roll=lidar_rot[0], pitch=lidar_rot[1], yaw=lidar_rot[2])
        )

        lidar_to_imu_se3 = PoseSE3(
            x=lidar_pos[0],
            y=-lidar_pos[1],  # Invert Y
            z=lidar_pos[2],
            qw=quaternion.qw,
            qx=quaternion.qx,
            qy=quaternion.qy,
            qz=quaternion.qz,
        )

        return LidarMetadata(
            lidar_name=str(lidar_id),
            lidar_id=lidar_id,
            lidar_to_imu_se3=lidar_to_imu_se3,
        )

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
            modalities_sync = self._convert_to_py123d(input_data, timestamp)
            self._py123d_log_writer.write_sync(modalities_sync)
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
    def _convert_to_py123d(self, input_data: dict, timestamp: float) -> ModalitiesSync:
        """Convert LEAD's processed data to Py123D format.

        Args:
            input_data: Sensor data from CARLA.
            timestamp: Current simulation timestamp in seconds.

        Returns:
            ModalitiesSync containing all modalities for this frame.
        """
        ts = Timestamp.from_s(timestamp)
        modalities: list[typing.Any] = []

        # Extract ego state
        ego_state = self._extract_py123d_ego_state(ts)
        modalities.append(ego_state)

        # Convert LEAD's bounding boxes format to Py123D
        box_detections = self._extract_py123d_box_detections(input_data, ts)
        LOG.debug(f"Extracted {len(box_detections.box_detections)} bounding boxes")
        if box_detections.box_detections:
            modalities.append(box_detections)

        # Convert camera data (needs ego state for world-space extrinsic)
        camera = self._extract_py123d_camera_data(input_data, ts, ego_state)
        LOG.debug("Extracted camera frame")
        modalities.append(camera)

        # Convert LiDAR data
        lidar = self._extract_py123d_lidar_data(ts)
        if lidar is not None:
            LOG.debug(f"Extracted LiDAR with {lidar.point_cloud_3d.shape[0]} points")
            modalities.append(lidar)

        # Traffic light detections
        traffic_lights = self._extract_py123d_traffic_lights(ts)
        modalities.append(traffic_lights)

        return ModalitiesSync(timestamp=ts, modalities=modalities)

    @beartype
    def _extract_py123d_ego_state(self, timestamp: Timestamp) -> EgoStateSE3:
        """Extract ego state from CARLA in Py123D format.

        Args:
            timestamp: Current simulation timestamp.

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

        return EgoStateSE3.from_center(
            center_se3=ego_center_se3,
            metadata=self._ego_metadata,
            timestamp=timestamp,
            dynamic_state_se3=dynamic_state,
        )

    @beartype
    def _extract_py123d_box_detections(
        self, input_data: dict, timestamp: Timestamp
    ) -> BoxDetectionsSE3:
        """Convert LEAD's bounding boxes format to Py123D.

        Args:
            input_data: Sensor data containing bounding boxes.
            timestamp: Current simulation timestamp.

        Returns:
            BoxDetectionsSE3 with all detected objects.
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
                    attributes=BoxDetectionAttributes(
                        label=label,
                        track_token=str(vehicle.id),
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
                    attributes=BoxDetectionAttributes(
                        label=DefaultBoxDetectionLabel.PERSON,
                        track_token=str(walker.id),
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
                    and bb["mesh_path"] is not None
                    and "Car" in bb["mesh_path"]
                ):
                    box_detections.append(
                        BoxDetectionSE3(
                            attributes=BoxDetectionAttributes(
                                label=DefaultBoxDetectionLabel.VEHICLE,
                                track_token=str(bb["id"]),
                            ),
                            bounding_box_se3=expert_py123d_utils.get_bounding_box_se3(
                                bb
                            ),
                            velocity_3d=expert_py123d_utils.get_bounding_box_velocity(
                                bb
                            ),
                        )
                    )
                    LOG.debug(f"Parking car detected: {bb['mesh_path']}")
                elif actor is None:  # static_prob_car that is not spawned as an actor
                    assert bb["class"] == "static_prop_car"
                    box_detections.append(
                        BoxDetectionSE3(
                            attributes=BoxDetectionAttributes(
                                label=DefaultBoxDetectionLabel.VEHICLE,
                                track_token=str(bb["id"]),
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
                                attributes=BoxDetectionAttributes(
                                    label=label,
                                    track_token=str(actor.id),
                                ),
                                bounding_box_se3=expert_py123d_utils.get_actor_bounding_box_se3(
                                    actor, overwrite_extent=overwrite_extent
                                ),
                                velocity_3d=expert_py123d_utils.get_actor_velocity(
                                    actor
                                ),
                            )
                        )

        return BoxDetectionsSE3(
            box_detections=box_detections,
            timestamp=timestamp,
            metadata=self._box_detections_metadata,
        )

    @beartype
    def _extract_py123d_camera_data(
        self, input_data: dict, timestamp: Timestamp, ego_state: EgoStateSE3
    ) -> Camera:
        """Extract camera data from LEAD's input_data.

        Args:
            input_data: Sensor data containing RGB images.
            timestamp: Current simulation timestamp.
            ego_state: Ego state for composing camera extrinsic in global frame.

        Returns:
            Camera with RGB image.
        """
        # Get RGB image from LEAD (CARLA provides BGRA, convert to RGB)
        bgra_image = (
            input_data["rgb_1"][1]
            if isinstance(input_data["rgb_1"], tuple)
            else input_data["rgb_1"]
        )
        rgb_image = bgra_image[:, :, :3][:, :, ::-1].copy()

        # Compose camera-to-global: imu_to_global @ camera_to_imu
        camera_to_global = rel_to_abs_se3(
            origin=ego_state.imu_se3,
            pose_se3=self._camera_metadata.camera_to_imu_se3,
        )

        return Camera(
            metadata=self._camera_metadata,
            image=rgb_image,
            camera_to_global_se3=camera_to_global,
            timestamp=timestamp,
        )

    @beartype
    def _extract_py123d_traffic_lights(
        self, timestamp: Timestamp
    ) -> TrafficLightDetections:
        """Extract traffic light detections from CARLA.

        Uses close_traffic_lights populated by the parent expert class and
        list_traffic_lights to resolve affected lane IDs.

        Args:
            timestamp: Current simulation timestamp.

        Returns:
            TrafficLightDetections with per-lane status.
        """
        detections: list[TrafficLightDetection] = []

        # Build actor → waypoints lookup from list_traffic_lights
        actor_waypoints: dict[int, list[carla.Waypoint]] = {
            actor.id: waypoints
            for actor, _center, waypoints in self.list_traffic_lights
        }

        for _tl_actor, _bb, tl_state, tl_id, _affects_ego in self.close_traffic_lights:
            status = expert_py123d_utils.carla_traffic_light_status_to_py123d(tl_state)
            waypoints = actor_waypoints.get(tl_id, [])
            if waypoints:
                for wp in waypoints:
                    detections.append(
                        TrafficLightDetection(lane_id=wp.lane_id, status=status)
                    )
            else:
                LOG.warning(
                    f"Traffic light ID {tl_id} has no associated waypoints. Using negative actor ID as lane_id fallback."
                )

        return TrafficLightDetections(detections=detections, timestamp=timestamp)

    @beartype
    def _extract_py123d_lidar_data(self, timestamp: Timestamp) -> Lidar | None:
        """Extract LiDAR data from accumulated point cloud.

        Args:
            timestamp: Current simulation timestamp.

        Returns:
            Lidar with point cloud, or None if no data.
        """
        lidar_pc = self.accumulate_lidar().copy()

        # Convert to ISO 8855: invert Y, shift X by rear axle offset, adjust Z
        lidar_pc[:, 1] = -lidar_pc[:, 1]  # Y
        lidar_pc[:, 0] += self._ego_metadata.rear_axle_to_center_longitudinal  # X
        lidar_pc[:, 2] += self.config_expert.lidar_pos_1[-1] / 2  # Z

        # Split into xyz (Nx3) and features
        point_cloud_3d = lidar_pc[:, :3].astype(np.float32)
        num_points = lidar_pc.shape[0]
        point_cloud_features = {
            LidarFeature.IDS.serialize(): np.full(
                num_points, int(LidarID.LIDAR_TOP.value), dtype=np.uint8
            ),
            LidarFeature.INTENSITY.serialize(): np.ones(num_points, dtype=np.uint8),
        }

        return Lidar(
            timestamp=timestamp,
            timestamp_end=timestamp,
            metadata=self._lidar_metadata,
            point_cloud_3d=point_cloud_3d,
            point_cloud_features=point_cloud_features,
        )

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
        if results is not None and self.save_path is not None:
            with open(
                os.path.join(
                    self._py123d_logs_root.absolute(),
                    self.config_expert.py123d_split,
                    self._log_name + ".json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results.__dict__, f, indent=2)
