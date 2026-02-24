# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Module-level storage for previous joint positions (keyed by env id)
_prev_joint_pos: dict[int, torch.Tensor] = {}


def joint_vel_from_pos_finite_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Estimate joint velocity via finite difference of consecutive joint positions.

    Computes: vel_est = (pos_current - pos_previous) / step_dt

    This provides a velocity signal derived purely from position measurements,
    suitable for sim2real on hardware without velocity sensors (e.g., SO-ARM101).

    On the first call or after environment reset, returns zeros (no velocity information yet).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_id = id(env)

    # Current relative joint positions
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # Get or initialize previous position buffer
    if env_id not in _prev_joint_pos or _prev_joint_pos[env_id].shape != current_pos.shape:
        # First call: initialize and return zeros
        _prev_joint_pos[env_id] = current_pos.clone()
        return torch.zeros_like(current_pos)

    # Compute finite difference velocity
    dt = env.step_dt
    vel_est = (current_pos - _prev_joint_pos[env_id]) / dt

    # Handle environment resets: zero out velocity for reset envs
    # After reset, the position jump is artificial, not real motion
    if hasattr(env, 'reset_buf'):
        reset_mask = env.reset_buf.bool()
        if reset_mask.any():
            vel_est[reset_mask] = 0.0

    # Store current position for next step
    _prev_joint_pos[env_id] = current_pos.clone()

    return vel_est


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b
