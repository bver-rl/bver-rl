from __future__ import annotations


from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class RandomActionNoiseCfg:
    """Correlated (OU) action noise for the random-env slice of the on-policy runner.

    Filters a white increment as ``state <- beta*state + (1-beta)*eps`` and outputs ``alpha*state``.
    """

    noise_type: str = "uniform"
    """White increment: ``"uniform"`` (legacy noise when ``beta`` is zero) or ``"ou"`` for a Gaussian increment."""

    beta: float = 0.0
    """Low-pass correlation coefficient in [0, 1]: zero is white noise, near one is heavy smoothing."""

    alpha: float = 1.0
    """Output scale applied to the filtered noise."""

    init_from_joint_pos: bool = True
    """Seed the filter on reset so the first joint target equals the reset pose, not the default stance.

    Requires the env to expose ``get_action_reset_state()`` and a non-zero ``alpha``.
    """

    persistent_seed: bool = False
    """Keep the seed as a constant offset so walks stay centered on the reset pose; needs ``init_from_joint_pos``."""


@configclass
# class ParkourPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
#     symmetry_cfg = RslRlSymmetryCfg(
#         use_data_augmentation=True,
#         use_mirror_loss=False,
#         data_augmentation_func=compute_symmetric_states,
#     )
class ParkourPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg = None


@configclass
class ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    train_env_ratio = 1.0
    random_env_ratio = 0.0
    # Base/fallback action noise, used by any slice below that doesn't set its own override.
    random_action_noise = RandomActionNoiseCfg()
    # Fraction of the random tail for reachability walks, split as [frontier | reachability].
    random_reachability_ratio = 0.0
    # Fraction of the frontier slice (not the whole tail) driving backward expansion; the rest is forward.
    random_backward_ratio = 0.0
    # Per-slice action noise; None reuses ``random_action_noise``.
    random_action_noise_forward: RandomActionNoiseCfg | None = None
    random_action_noise_backward: RandomActionNoiseCfg | None = None
    random_action_noise_reachability: RandomActionNoiseCfg | None = None
    max_iterations = 500
    save_interval = 100
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=2.0,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = ParkourPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalDClimbBoxPPORunnerCfg(ParkourPPORunnerCfg):
    experiment_name = "anymal_d_climb_box"
    max_iterations = 2000


@configclass
class AnymalDClimbBoxRCPPORunnerCfg(AnymalDClimbBoxPPORunnerCfg):
    train_env_ratio = 0.9
    random_env_ratio = 0.1
    # Legacy-equivalent white noise.
    random_action_noise = RandomActionNoiseCfg(noise_type="uniform", beta=0.0, alpha=1.0)


@configclass
class AnymalDClimbBoxBVERPPORunnerCfg(AnymalDClimbBoxPPORunnerCfg):
    """Bidirectional BVER with a three-way tail split ``[forward | backward | reachability]``.

    Pairs with ``ClimbBoxBVERCurriculumCfg``. The commented entropy pin is a fallback if std never anneals.
    """

    train_env_ratio = 0.8
    random_env_ratio = 0.2
    random_reachability_ratio = 0.25
    random_backward_ratio = 0.5
    # random_action_noise_forward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0)
    # random_action_noise_backward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0)
    # random_action_noise_reachability = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0)
    random_action_noise_forward = RandomActionNoiseCfg(noise_type="uniform", beta=0.95, alpha=10.0)
    random_action_noise_backward = RandomActionNoiseCfg(noise_type="uniform", beta=0.95, alpha=10.0)
    random_action_noise_reachability = RandomActionNoiseCfg(noise_type="uniform", beta=0.95, alpha=10.0)

    # def __post_init__(self):
    #     self.algorithm.entropy_coef = 0.001


@configclass
class AnymalDClimbBoxBVERNoReachPPORunnerCfg(AnymalDClimbBoxBVERPPORunnerCfg):
    """Bidirectional BVER without reachability; the freed envs return to training.

    Pairs with ``ClimbBoxBVERNoReachCurriculumCfg``, which asserts ``random_reachability_ratio`` is zero.
    """

    train_env_ratio = 0.85
    random_env_ratio = 0.15
    random_reachability_ratio = 0.0


@configclass
class AnymalDClimbBoxBVERSparsePPORunnerCfg(AnymalDClimbBoxBVERNoReachPPORunnerCfg):
    """Bidirectional sparse-reward BVER arm (``climb_single_box_cfg_bver_sparse.py``) with OU walk noise."""

    experiment_name = "anymal_d_climb_box_sparse"

    random_action_noise_forward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)
    random_action_noise_backward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)


@configclass
class AnymalDClimbBoxBVERSparseFwdPPORunnerCfg(AnymalDClimbBoxBVERSparsePPORunnerCfg):
    """Forward-only sparse-reward BVER arm; pairs with ``ClimbBoxBVERSparseFwdCurriculumCfg``."""

    random_backward_ratio = 0.0


@configclass
class AnymalDClimbBoxBVERSparseBwdPPORunnerCfg(AnymalDClimbBoxBVERSparsePPORunnerCfg):
    """Backward-only sparse-reward BVER arm; pairs with ``ClimbBoxBVERSparseBwdCurriculumCfg``."""

    random_backward_ratio = 1.0


@configclass
class AnymalDClimbBoxSparsePPORunnerCfg(AnymalDClimbBoxPPORunnerCfg):
    """Plain-PPO runner for the sparse controls without a random tail (Baseline, Random, RSI, Backplay).

    Every env trains; RC uses ``AnymalDClimbBoxRCSparsePPORunnerCfg`` instead.
    """

    experiment_name = "anymal_d_climb_box_sparse"


@configclass
class AnymalDClimbBoxRCSparsePPORunnerCfg(AnymalDClimbBoxRCPPORunnerCfg):
    """Reverse-curriculum control of the sparse campaign, on the dense RC tail split."""

    experiment_name = "anymal_d_climb_box_sparse"

    # random_action_noise = RandomActionNoiseCfg(noise_type="uniform", beta=0.0, alpha=2.77)
    # random_action_noise = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)
    random_action_noise = RandomActionNoiseCfg(noise_type="ou", beta=0.0, alpha=1.6)


@configclass
class AnymalDClimbDownBoxPPORunnerCfg(ParkourPPORunnerCfg):
    experiment_name = "anymal_d_climb_down_box"


    # random_action_noise = RandomActionNoiseCfg(noise_type="uniform", beta=0.0, alpha=1.0)


@configclass
class AnymalDClimbDownBoxBVERPPORunnerCfg(AnymalDClimbDownBoxPPORunnerCfg):
    """Climb-down mirror of ``AnymalDClimbBoxBVERPPORunnerCfg`` with its own log dir."""

    train_env_ratio = 0.8
    random_env_ratio = 0.2
    random_reachability_ratio = 0.25
    random_backward_ratio = 0.5
    random_action_noise_forward = RandomActionNoiseCfg(noise_type="uniform", beta=0.95, alpha=10.0)
    random_action_noise_backward = RandomActionNoiseCfg(noise_type="uniform", beta=0.95, alpha=10.0)
    random_action_noise_reachability = RandomActionNoiseCfg(noise_type="uniform", beta=0.95, alpha=10.0)


@configclass
class AnymalDClimbDownBoxBVERNoReachPPORunnerCfg(AnymalDClimbDownBoxBVERPPORunnerCfg):
    """Climb-down mirror of ``AnymalDClimbBoxBVERNoReachPPORunnerCfg``."""

    train_env_ratio = 0.85
    random_env_ratio = 0.15
    random_reachability_ratio = 0.0


@configclass
class AnymalDClimbDownBoxBVERSparsePPORunnerCfg(AnymalDClimbDownBoxBVERNoReachPPORunnerCfg):
    """Climb-down mirror of ``AnymalDClimbBoxBVERSparsePPORunnerCfg`` with its own log tree."""

    experiment_name = "anymal_d_climb_down_box_sparse"

    random_action_noise_forward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)
    random_action_noise_backward = RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0)


@configclass
class AnymalDClimbDownBoxBVERSparseFwdPPORunnerCfg(AnymalDClimbDownBoxBVERSparsePPORunnerCfg):
    """Forward-only sparse-reward BVER arm, climb-down."""

    random_backward_ratio = 0.0


@configclass
class AnymalDClimbDownBoxBVERSparseBwdPPORunnerCfg(AnymalDClimbDownBoxBVERSparsePPORunnerCfg):
    """Backward-only sparse-reward BVER arm, climb-down."""

    random_backward_ratio = 1.0


