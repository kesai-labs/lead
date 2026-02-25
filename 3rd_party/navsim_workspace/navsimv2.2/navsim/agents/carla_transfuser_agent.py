from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
import logging

from open_loop_inference import OpenLoopInference as CarlaOpenLoopInference
from open_loop_inference import OpenLoopPrediction as CarlaOpenLoopPrediction
import numpy as np
from config_open_loop import OpenLoopConfig as CarlaOpenLoopConfig
from config_training import TrainingConfig as CarlaTrainingConfig
import random
import cv2
import json
import os

logger =logging.getLogger(__name__)

class CarlaTransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""
    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: str,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent.
        :param checkpoint_path: optional path string to checkpoint, defaults to None.
        """
        super().__init__(trajectory_sampling)

        self._config = config
        
        logger.info(f"Loading CARLA TransFuser model from input path: {checkpoint_path}")
        self._checkpoint_path = checkpoint_path.rsplit("/", 1)[0]
        self._model_filename = checkpoint_path.rsplit("/", 1)[1]
        logger.info(f"Model filename: {self._model_filename}")
        logger.info(f"Checkpoint path: {self._checkpoint_path}")

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        with open(os.path.join(self._checkpoint_path, "config.json"), "r", encoding="utf-8") as f:
            json_config = json.load(f)
        self._carla_model_config = CarlaTrainingConfig(json_config)
        self._carla_config_open_loop = CarlaOpenLoopConfig()
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            idx = random.randint(0, gpu_count - 1)
            device = torch.device(f"cuda:{idx}")
        elif gpu_count == 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self._carla_open_loop_inference = CarlaOpenLoopInference(
            config_training=self._carla_model_config,
            config_open_loop=self._carla_config_open_loop,
            model_path=self._checkpoint_path,
            device=device,
            prefix=self._model_filename
        )

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: Transfuser only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=history_steps,
            lidar_pc=False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
    
        rgb : torch.Tensor = features["camera_feature"]
        rgb = rgb.numpy()
        rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC to CHW
        rgb = torch.tensor(rgb).unsqueeze(0).float()  # CHW to NCHW

        output: CarlaOpenLoopPrediction = self._carla_open_loop_inference.forward({
            "rgb": rgb,
            "command": features["status_feature"][:, :4].reshape(-1, 4),
            "speed": torch.linalg.norm(features["status_feature"][:, 4:6]).reshape(-1, 1),
            "acceleration": torch.linalg.norm(features["status_feature"][:, 6:8]).reshape(-1, 1),
        })

        future_waypoints = output.pred_future_waypoints.clone()

        # Model trained in mixed dataset, need to flip y axis
        future_waypoints[:, :, 1] = -future_waypoints[:, :, 1]
        future_headings = -output.pred_future_headings.clone()

        return {
            "trajectory": torch.concatenate([future_waypoints, future_headings.unsqueeze(-1)], dim=-1).cpu()
        }

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("CARLA TransFuser supports only inference.")

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        raise NotImplementedError("CARLA TransFuser supports only inference.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        raise NotImplementedError("CARLA TransFuser supports only inference.")
