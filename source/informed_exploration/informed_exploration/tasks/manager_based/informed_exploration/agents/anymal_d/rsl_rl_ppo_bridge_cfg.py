"""RSL-RL PPO runner configs for the bridge terrains (see `informed_exploration.terrains`).

The BVER arms mirror the climb-box no-reach ratios.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import ParkourPPORunnerCfg, RandomActionNoiseCfg


@configclass
class AnymalDBridgeDTSGPPORunnerCfg(ParkourPPORunnerCfg):
    experiment_name = "anymal_d_bridge_dtsg"
    max_iterations = 4000


@configclass
class AnymalDBridgeDTSGHighEntropyPPORunnerCfg(AnymalDBridgeDTSGPPORunnerCfg):
    """Baseline arm with a raised entropy bonus, its only exploration lever."""

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.01


@configclass
class AnymalDBridgeDTSGBVERPPORunnerCfg(AnymalDBridgeDTSGPPORunnerCfg):
    """Bidirectional no-reachability BVER; pairs with ``BridgeBVERCurriculumCfg`` on the DTSG terrain."""

    train_env_ratio = 0.85
    random_env_ratio = 0.15
    random_reachability_ratio = 0.0
    random_backward_ratio = 0.5
    random_action_noise_forward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)
    random_action_noise_backward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)


@configclass
class AnymalDBridgeDTSGBVERFwdPPORunnerCfg(AnymalDBridgeDTSGBVERPPORunnerCfg):
    """Forward-only ablation arm. Pairs with ``BridgeBVERFwdCurriculumCfg``."""

    random_backward_ratio = 0.0


@configclass
class AnymalDBridgeDTSGBVERBwdPPORunnerCfg(AnymalDBridgeDTSGBVERPPORunnerCfg):
    """Backward-only ablation arm. Pairs with ``BridgeBVERBwdCurriculumCfg``."""

    random_backward_ratio = 1.0


@configclass
class AnymalDBridgeDTSGRandomGSPPORunnerCfg(AnymalDBridgeDTSGPPORunnerCfg):
    """Uninformed random (start, goal) control for all three ``BridgeRandomGS*CurriculumCfg`` arms.

    The env split matches the BVER runner so the PPO batch is equal; the unused tail is deliberate.
    """

    train_env_ratio = 0.85
    random_env_ratio = 0.15
    random_reachability_ratio = 0.0
    random_backward_ratio = 0.5
    random_action_noise_forward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0)
    random_action_noise_backward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0)


