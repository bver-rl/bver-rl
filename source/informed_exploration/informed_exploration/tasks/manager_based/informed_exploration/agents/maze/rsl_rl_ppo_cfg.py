# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from isaaclab.utils import configclass

from ..anymal_d.rsl_rl_ppo_cfg import ParkourPPOAlgorithmCfg, ParkourPPORunnerCfg, RandomActionNoiseCfg


@configclass
class MazePPORunnerCfg(ParkourPPORunnerCfg):
    """PPO for the point-mass maze, bound by every arm without a random-action tail.

    Observation normalization is off since the noise-free obs are already in metres and metres per second.
    """

    experiment_name = "maze"
    num_steps_per_env = 16

    # Leftover envs are held out of the PPO update and report the headline task live.
    train_env_ratio = 0.9
    max_iterations = 1000
    save_interval = 100

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=4.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64, 16],
        critic_hidden_dims=[128, 64, 16],
        activation="elu",
    )
    algorithm = ParkourPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MazeBVERPPORunnerCfg(MazePPORunnerCfg):
    """Bidirectional BVER without reachability, which the curriculum asserts.

    Walk noise is strongly correlated, since uncorrelated velocity actions cover almost no ground.
    """

    train_env_ratio = 0.7
    random_env_ratio = 0.2
    random_reachability_ratio = 0.0
    random_backward_ratio = 0.5

    random_action_noise_forward = RandomActionNoiseCfg(
        noise_type="ou", beta=0.95, alpha=10.0, init_from_joint_pos=False
    )
    random_action_noise_backward = RandomActionNoiseCfg(
        noise_type="ou", beta=0.95, alpha=10.0, init_from_joint_pos=False
    )
    random_action_noise_reachability = RandomActionNoiseCfg(
        noise_type="ou", beta=0.95, alpha=10.0, init_from_joint_pos=False
    )


@configclass
class MazeBVERFwdPPORunnerCfg(MazeBVERPPORunnerCfg):
    """Forward-only: the whole frontier slice walks forward."""

    random_backward_ratio = 0.0


@configclass
class MazeBVERBwdPPORunnerCfg(MazeBVERPPORunnerCfg):
    """Backward-only: the whole frontier slice walks backward."""

    random_backward_ratio = 1.0
