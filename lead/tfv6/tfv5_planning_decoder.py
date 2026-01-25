import math

import jaxtyping as jt
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

from lead.common import common_utils as t_u
from lead.training.config_training import TrainingConfig


class TFv5PlanningDecoder(nn.Module):
    @beartype
    def __init__(
        self, input_bev_channels: int, config: TrainingConfig, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.planning_context_encoder = PlanningContextEncoder(
            config=self.config,
            input_bev_channels=input_bev_channels,
            device=self.device,
        )

        self.query = nn.Parameter(
            torch.zeros(
                1,
                self.config.num_route_points_prediction
                + self.config.num_way_points_prediction
                + 1,
                self.config.transfuser_token_dim,
            )
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.config.transfuser_token_dim,
                self.config.transfuser_num_bev_cross_attention_heads,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=self.config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(self.config.transfuser_token_dim),
        )

        self.tp_encoder = nn.Linear(2, config.gru_hidden_size)

        self.route_gru = torch.nn.GRU(
            input_size=config.transfuser_token_dim,
            hidden_size=config.gru_hidden_size,
            batch_first=True,
        )
        self.route_decoder = nn.Sequential(nn.Linear(config.gru_hidden_size, 2))

        self.wp_gru = torch.nn.GRU(
            input_size=config.transfuser_token_dim,
            hidden_size=config.gru_hidden_size,
            batch_first=True,
        )
        self.wp_decoder = nn.Sequential(nn.Linear(config.gru_hidden_size, 2))

        self.target_speed_decoder = nn.Sequential(
            nn.Linear(
                self.config.transfuser_token_dim,
                self.config.transfuser_token_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.transfuser_token_dim, len(self.config.target_speeds)),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query)

    @beartype
    def forward(
        self,
        bev_features: torch.Tensor,
        radar_logits: jt.Float[torch.Tensor, "B Q C"] | None,
        radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None,
        data: dict,
        log: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Args:
            bev_features: [bs, input_dim, height, width]
            radar_logits: radar classification logits (not used)
            radar_predictions: radar box predictions (not used)
            data: dict
            log: dict
        Returns:
            route: [bs, num_route_points, 2]
            waypoints: [bs, num_way_points, 2]
            target_speed: [bs, len(target_speeds)] Target speed distribution
            target_speed_scalar: [bs,] Target speed in m/s
        """
        context_tokens = self.planning_context_encoder(
            bev_features=bev_features, data=data, log=log
        )

        bs = context_tokens.shape[0]

        query = self.query.repeat(bs, 1, 1)
        values = self.transformer_decoder(query, context_tokens)

        # Split the queries
        route_values, waypoints_values, speed_value = (
            values[:, : self.config.num_route_points_prediction],
            values[:, self.config.num_route_points_prediction : -1],
            values[:, -1],
        )
        target_point = data["target_point"].to(
            device=self.device, dtype=self.config.torch_float_type, non_blocking=True
        )
        hidden_state = self.tp_encoder(target_point).unsqueeze(0)

        # Forward-autoregression
        route_values, _ = self.route_gru(route_values, hidden_state)
        waypoints_values, _ = self.wp_gru(waypoints_values, hidden_state)

        route = torch.cumsum(self.route_decoder(route_values), 1)
        waypoints = torch.cumsum(self.wp_decoder(waypoints_values), 1)
        target_speed_dist = self.target_speed_decoder(speed_value)

        return (
            route,
            waypoints,
            target_speed_dist,
            decode_two_hot(torch.softmax(target_speed_dist, dim=-1), self.config),
            None,
        )

    @beartype
    def compute_loss(self, predictions, data: dict, loss: dict, log: dict):
        waypoints_label = data["future_waypoints"].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )[:, : self.config.num_way_points_prediction]
        target_speed_distribution = encode_two_hot(
            data["target_speed"].to(
                self.device, dtype=self.config.torch_float_type, non_blocking=True
            ),
            data["brake"].to(self.device, dtype=torch.bool, non_blocking=True),
            self.config,
        )
        route_label = data["route"].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )

        loss.update(
            {
                "loss_tf": F.l1_loss(
                    predictions.pred_future_waypoints, waypoints_label, reduction="none"
                ).mean(),
                "loss_tfpp_speed": F.cross_entropy(
                    predictions.pred_target_speed_distribution,
                    target_speed_distribution,
                ),
                "loss_tfpp": F.l1_loss(predictions.pred_route, route_label),
            }
        )

        if (
            "iteration" in data
            and ((data["iteration"] + 1) % self.config.wandb_log_scalars_frequency) == 0
        ):
            target_speed_labels = decode_two_hot(target_speed_distribution, self.config)
            log.update(
                {
                    "metric/route_ade": t_u.average_displacement_error(
                        predictions.pred_route, route_label
                    ),
                    "metric/route_fde": t_u.final_displacement_error(
                        predictions.pred_route, route_label
                    ),
                    "metric/target_speed_error": torch.mean(
                        torch.abs(
                            predictions.pred_target_speed_scalar - target_speed_labels
                        )
                    ).item(),
                    "metric/target_speed_correlation": torch.corrcoef(
                        torch.stack(
                            [predictions.pred_target_speed_scalar, target_speed_labels]
                        )
                    )[0, 1].item(),
                    "metric/waypoints_ade": t_u.average_displacement_error(
                        predictions.pred_future_waypoints, waypoints_label
                    ),
                    "metric/waypoints_fde": t_u.final_displacement_error(
                        predictions.pred_future_waypoints, waypoints_label
                    ),
                }
            )
            log["metric/tfpp_speed_error"] = log["metric/target_speed_error"]
            log["metric/tfpp_ade"] = log["metric/route_ade"]
            log["metric/tfpp_fde"] = log["metric/route_fde"]
            log["metric/tf_ade"] = log["metric/waypoints_ade"]
            log["metric/tf_fde"] = log["metric/waypoints_fde"]


@beartype
def decode_two_hot(
    two_hot_label: jt.Float[torch.Tensor, "B C"], config: TrainingConfig
) -> jt.Float[torch.Tensor, " B"]:
    """Decode a two-hot encoded tensor into a scalar representation.

    Args:
        two_hot_label: The two-hot encoded tensor. Must be between 0 and 1 and sum to 1 along the last dimension.
        config: The training configuration.

    Returns:
        jt.Float[torch.Tensor, " B"]: The decoded scalar tensor.
    """
    assert torch.all((0.0 <= two_hot_label) & (two_hot_label <= 1.0)), (
        "Two-hot labels must be between 0 and 1"
    )
    assert torch.allclose(
        torch.sum(two_hot_label, dim=-1),
        torch.ones(
            two_hot_label.shape[0],
            device=two_hot_label.device,
            dtype=two_hot_label.dtype,
        ),
    ), "Two-hot labels must sum to 1"

    target_speeds = torch.tensor(
        config.target_speeds, device=two_hot_label.device, dtype=two_hot_label.dtype
    ).unsqueeze(0)
    target_speed = (two_hot_label * target_speeds).sum(axis=-1)
    return target_speed


@beartype
def encode_two_hot(
    scalar_speed: jt.Float[torch.Tensor, " B"],
    brake: jt.Bool[torch.Tensor, " B"],
    config: TrainingConfig,
) -> jt.Float[torch.Tensor, "B C"]:
    assert all(scalar_speed >= 0.0)
    target_speeds = torch.tensor(
        config.target_speeds, dtype=scalar_speed.dtype, device=scalar_speed.device
    )
    labels = torch.zeros(
        len(scalar_speed),
        len(target_speeds),
        dtype=scalar_speed.dtype,
        device=scalar_speed.device,
    )
    labels[brake, 0] = 1.0
    non_brake = ~brake
    speeds = scalar_speed[non_brake]
    last_bin = speeds >= target_speeds[-1]
    labels[non_brake & (scalar_speed >= target_speeds[-1]), -1] = 1.0

    # Interpolation between bins
    interp_mask = ~last_bin
    if interp_mask.any():
        interp_speeds = speeds[interp_mask]
        upper_idx = torch.searchsorted(target_speeds, interp_speeds, right=False)
        lower_idx = upper_idx - 1

        lower_val = target_speeds[lower_idx]
        upper_val = target_speeds[upper_idx]

        lower_weight = (upper_val - interp_speeds) / (upper_val - lower_val)
        upper_weight = (interp_speeds - lower_val) / (upper_val - lower_val)

        row_idx = torch.where(non_brake)[0][interp_mask]
        labels[row_idx, lower_idx] = lower_weight
        labels[row_idx, upper_idx] = upper_weight

    return labels


class PlanningContextEncoder(nn.Module):
    @beartype
    def __init__(
        self, config: TrainingConfig, input_bev_channels: int, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.config = config

        self.num_status_tokens = 0

        if self.config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            print("Model will use velocity.")

        if self.config.use_discrete_command:
            self.num_status_tokens += 1
            self.command_encoder = nn.Sequential(
                nn.Linear(6, self.config.transfuser_token_dim)
            )
            print("Model will use discrete command.")

        self.cosine_pos_embeding = PositionEmbeddingSine(
            config, self.config.transfuser_token_dim // 2, normalize=True
        )
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_status_tokens, self.config.transfuser_token_dim)
        )

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels, self.config.transfuser_token_dim, kernel_size=1
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.status_pos_embedding)

    @beartype
    def forward(
        self, bev_features: jt.Float[torch.Tensor, "B C H W"], data: dict, log: dict
    ) -> jt.Float[torch.Tensor, "B N D"]:
        """
        Args:
            bev_features: [bs, input_dim, height, width]
            data: dict
            log: dict
        Returns:
            context_tokens: [bs, num_context_tokens, transfuser_token_dim]
        """
        # Load data
        if self.config.use_velocity:
            velocity = (
                data["speed"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
        if self.config.use_discrete_command:
            command = data["command"].to(
                self.device, dtype=self.config.torch_float_type
            )

        status_tokens = []

        # Encode speed
        if self.config.use_velocity:
            velocity_token = self.velocity_encoder(
                velocity / self.config.max_speed
            ).reshape(
                -1, 1, self.config.transfuser_token_dim
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(velocity_token)

        # Encode command
        if self.config.use_discrete_command:
            command_token = self.command_encoder(command).reshape(
                -1, 1, self.config.transfuser_token_dim
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(command_token)

        # Concatenate status tokens if any
        has_statuses = False
        if len(status_tokens) > 0:
            status_tokens = torch.cat(
                status_tokens, dim=1
            )  # (bs, num_status_tokens, transfuser_token_dim)
            has_statuses = True

        # Process BEV features
        context_tokens = self.dimension_adapter(
            bev_features
        )  # (bs, transfuser_token_dim, height, width)

        # Concatenate and add positional embeddings
        if has_statuses:
            context_tokens = context_tokens + self.cosine_pos_embeding(
                context_tokens
            )  # (bs, transfuser_token_dim, height, width)
            context_tokens = torch.flatten(
                context_tokens, start_dim=2
            )  # (bs, transfuser_token_dim, height * width)
            context_tokens = torch.permute(
                context_tokens, (0, 2, 1)
            )  # (bs, height * width, transfuser_token_dim)

            status_tokens = (
                status_tokens + self.status_pos_embedding
            )  # (bs, num_status_tokens, transfuser_token_dim)
            context_tokens = torch.cat(
                [context_tokens, status_tokens], dim=1
            )  # (bs, height * width + num_status_tokens, transfuser_token_dim)

        return context_tokens


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.config = config
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor: torch.Tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(self.config.torch_float_type).contiguous()
