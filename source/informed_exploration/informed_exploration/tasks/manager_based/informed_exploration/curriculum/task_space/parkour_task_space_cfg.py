import math

from isaaclab.utils import configclass

height_offset = 0.6

jpl = joint_position_limit = math.pi / 2
jvl = joint_velocity_limit = math.pi / 18

# pos_only_bounds: list | None = [
#     # base pose and velocity
#     [-3.5, 0.0],  # x
#     [0.0, 0.0],  # vx
#     [-1.0, 1.0],  # y
#     [0.0, 0.0],  # vy
#     [0.6, 1.2],  # z
#     [0.0, 0.0],  # vz # TODO: change this to -1.0
#     [0.0, 0.0],  # roll
#     [0.0, 0.0],  # v roll
#     [0.0, 0.0],  # pitch
#     [0.0, 0.0],  # v pitch
#     [0.0, 0.0],  # yaw
#     [0.0, 0.0],  # v yaw
#     # joint positions and velocities
#     # Hip abduction/adduction
#     [0.0, 0.0],  # j1 pos LF_HAA
#     [0.0, 0.0],  # j1 vel
#     [0.0, 0.0],  # j2 pos LH_HAA
#     [0.0, 0.0],  # j2 vel
#     [0.0, 0.0],  # j3 pos RF_HAA
#     [0.0, 0.0],  # j3 vel
#     [0.0, 0.0],  # j4 pos RH_HAA
#     [0.0, 0.0],  # j4 vel
#     # Hip flexion/extension
#     [0.0, 0.0],  # j5 pos LF_HFE
#     [0.0, 0.0],  # j5 vel
#     [0.0, 0.0],  # j6 pos LH_HFE
#     [0.0, 0.0],  # j6 vel
#     [0.0, 0.0],  # j7 pos RF_HFE
#     [0.0, 0.0],  # j7 vel
#     [0.0, 0.0],  # j8 pos RH_HFE
#     [0.0, 0.0],  # j8 vel
#     # Knee flexion/extension
#     [0.0, 0.0],  # j9 pos LF_KFE
#     [0.0, 0.0],  # j9 vel
#     [0.0, 0.0],  # j10 pos LH_KFE
#     [0.0, 0.0],  # j10 vel
#     [0.0, 0.0],  # j11 pos RF_KFE
#     [0.0, 0.0],  # j11 vel
#     [0.0, 0.0],  # j12 pos RH_KFE
#     [0.0, 0.0],  # j12 vel
# ]

# pose_only_bounds: list | None = [
#     # base pose and velocity
#     [-3.5, 0.0],  # x
#     [-1.0, 1.0],  # vx
#     [-1.0, 1.0],  # y
#     [-1.0, 1.0],  # vy
#     [0.6, 1.2],  # z
#     [0.0, 1.0],  # vz # TODO: change this to -1.0
#     [-math.pi / 6, math.pi / 6],  # roll
#     [-math.pi / 18, math.pi / 18],  # v roll
#     [-math.pi / 4, math.pi / 4],  # pitch
#     [-math.pi / 18, math.pi / 18],  # v pitch
#     [-math.pi, math.pi],  # yaw
#     [-math.pi / 18, math.pi / 18],  # v yaw
#     # joint positions and velocities
#     # Hip abduction/adduction
#     [0.0, 0.0],  # j1 pos LF_HAA
#     [0.0, 0.0],  # j1 vel
#     [0.0, 0.0],  # j2 pos LH_HAA
#     [0.0, 0.0],  # j2 vel
#     [0.0, 0.0],  # j3 pos RF_HAA
#     [0.0, 0.0],  # j3 vel
#     [0.0, 0.0],  # j4 pos RH_HAA
#     [0.0, 0.0],  # j4 vel
#     # Hip flexion/extension
#     [0.0, 0.0],  # j5 pos LF_HFE
#     [0.0, 0.0],  # j5 vel
#     [0.0, 0.0],  # j6 pos LH_HFE
#     [0.0, 0.0],  # j6 vel
#     [0.0, 0.0],  # j7 pos RF_HFE
#     [0.0, 0.0],  # j7 vel
#     [0.0, 0.0],  # j8 pos RH_HFE
#     [0.0, 0.0],  # j8 vel
#     # Knee flexion/extension
#     [0.0, 0.0],  # j9 pos LF_KFE
#     [0.0, 0.0],  # j9 vel
#     [0.0, 0.0],  # j10 pos LH_KFE
#     [0.0, 0.0],  # j10 vel
#     [0.0, 0.0],  # j11 pos RF_KFE
#     [0.0, 0.0],  # j11 vel
#     [0.0, 0.0],  # j12 pos RH_KFE
#     [0.0, 0.0],  # j12 vel
# ]

full_bounds: list | None = [
    # base pose and velocity
    [-3.5, 0.25],  # x
    [-1.0, 1.0],  # vx
    [-1.0, 1.0],  # y
    [-0.5, 0.5],  # vy
    [0.5, 1.2],  # z
    [0.0, 1.0],  # vz
    [-math.pi / 6, math.pi / 6],  # roll
    [-math.pi / 18, math.pi / 18],  # v roll
    [-math.pi / 4, math.pi / 4],  # pitch
    [-math.pi / 18, math.pi / 18],  # v pitch
    [-math.pi / 2, math.pi / 2],  # yaw
    [-math.pi / 18, math.pi / 18],  # v yaw
    # joint positions and velocities, hip abduction/adduction
    [-jpl, jpl],  # j1 pos LF_HAA
    [-jvl, jvl],  # j1 vel
    [-jpl, jpl],  # j2 pos LH_HAA
    [-jvl, jvl],  # j2 vel
    [-jpl, jpl],  # j3 pos RF_HAA
    [-jvl, jvl],  # j3 vel
    [-jpl, jpl],  # j4 pos RH_HAA
    [-jvl, jvl],  # j4 vel
    # Hip flexion/extension
    [-jpl, jpl],  # j5 pos LF_HFE
    [-jvl, jvl],  # j5 vel
    [-jpl, jpl],  # j6 pos LH_HFE
    [-jvl, jvl],  # j6 vel
    [-jpl, jpl],  # j7 pos RF_HFE
    [-jvl, jvl],  # j7 vel
    [-jpl, jpl],  # j8 pos RH_HFE
    [-jvl, jvl],  # j8 vel
    # Knee flexion/extension
    [-jpl, jpl],  # j9 pos LF_KFE
    [-jvl, jvl],  # j9 vel
    [-jpl, jpl],  # j10 pos LH_KFE
    [-jvl, jvl],  # j10 vel
    [-jpl, jpl],  # j11 pos RF_KFE
    [-jvl, jvl],  # j11 vel
    [-jpl, jpl],  # j12 pos RH_KFE
    [-jvl, jvl],  # j12 vel
]

# expert_bounds: list | None = [
#     # base pose and velocity
#     [-3.5, 0.0],  # x
#     [-0.0, 1.0],  # vx
#     [-1.0, 1.0],  # y
#     [0.0, 0.0],  # vy
#     [0.6, 1.2],  # z
#     [0.0, 2.0],  # vz # TODO: change this to -1.0
#     [0.0, 0.0],  # roll
#     [0.0, 0.0],  # v roll
#     [-math.pi / 3, 0.0],  # pitch
#     [0.0, 0.0],  # v pitch
#     [0.0, 0.0],  # yaw
#     [0.0, 0.0],  # v yaw
#     # joint positions and velocities
#     # Hip abduction/adduction
#     [0.0, 0.0],  # j1 pos LF_HAA
#     [0.0, 0.0],  # j1 vel
#     [0.0, 0.0],  # j2 pos LH_HAA
#     [0.0, 0.0],  # j2 vel
#     [0.0, 0.0],  # j3 pos RF_HAA
#     [0.0, 0.0],  # j3 vel
#     [0.0, 0.0],  # j4 pos RH_HAA
#     [0.0, 0.0],  # j4 vel
#     # Hip flexion/extension
#     [0.0, 0.0],  # j5 pos LF_HFE
#     [0.0, 0.0],  # j5 vel
#     [0.0, 0.0],  # j6 pos LH_HFE
#     [0.0, 0.0],  # j6 vel
#     [0.0, 0.0],  # j7 pos RF_HFE
#     [0.0, 0.0],  # j7 vel
#     [0.0, 0.0],  # j8 pos RH_HFE
#     [0.0, 0.0],  # j8 vel
#     # Knee flexion/extension
#     [0.0, 0.0],  # j9 pos LF_KFE
#     [0.0, 0.0],  # j9 vel
#     [0.0, 0.0],  # j10 pos LH_KFE
#     [0.0, 0.0],  # j10 vel
#     [0.0, 0.0],  # j11 pos RF_KFE
#     [0.0, 0.0],  # j11 vel
#     [0.0, 0.0],  # j12 pos RH_KFE
#     [0.0, 0.0],  # j12 vel
# ]

jw = joint_weights = 0.25

STATE_SPACE_WEIGHTS = [
    4.0,
    0.0,
    2.0,
    0.0,
    0.5,
    0.0,
    0.5,
    0.0,
    0.5,
    0.0,
    0.5,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
    jw,
    0.0,
]


@configclass
class ParkourTaskSpaceCfg:

    box_height: float = 0.8

    # (nr. of params, dimension of each param)
    dimensions: tuple | None = (18, 2)
    eval_dimensions: tuple | None = (18, 2)

    xyz_dimensions: tuple | None = (0, 2, 4)

    total_dims: int | None = dimensions[0] * dimensions[1] if dimensions is not None else None

    bounds: list | None = full_bounds

    bounds[4][1] = (
        height_offset + box_height
    )  # overwrite z max to be closer to target box height for sampling efficiency

    eval_bounds: list | None = [
        # base pose and velocity
        [-3.5, -2.5],  # x
        [0.0, 0.0],  # vx
        [-1.0, 1.0],  # y
        [0.0, 0.0],  # vy
        [0.6, 0.6],  # z
        [0.0, 0.0],  # vz
        [0.0, 0.0],  # roll
        [0.0, 0.0],  # v roll
        [0.0, 0.0],  # pitch
        [0.0, 0.0],  # v pitch
        [0.0, 0.0],  # yaw
        [0.0, 0.0],  # v yaw
        # joint positions and velocities, hip abduction/adduction
        [0.0, 0.0],  # j1 pos LF_HAA
        [0.0, 0.0],  # j1 vel
        [0.0, 0.0],  # j2 pos LH_HAA
        [0.0, 0.0],  # j2 vel
        [0.0, 0.0],  # j3 pos RF_HAA
        [0.0, 0.0],  # j3 vel
        [0.0, 0.0],  # j4 pos RH_HAA
        [0.0, 0.0],  # j4 vel
        # Hip flexion/extension
        [0.0, 0.0],  # j5 pos LF_HFE
        [0.0, 0.0],  # j5 vel
        [0.0, 0.0],  # j6 pos LH_HFE
        [0.0, 0.0],  # j6 vel
        [0.0, 0.0],  # j7 pos RF_HFE
        [0.0, 0.0],  # j7 vel
        [0.0, 0.0],  # j8 pos RH_HFE
        [0.0, 0.0],  # j8 vel
        # Knee flexion/extension
        [0.0, 0.0],  # j9 pos LF_KFE
        [0.0, 0.0],  # j9 vel
        [0.0, 0.0],  # j10 pos LH_KFE
        [0.0, 0.0],  # j10 vel
        [0.0, 0.0],  # j11 pos RF_KFE
        [0.0, 0.0],  # j11 vel
        [0.0, 0.0],  # j12 pos RH_KFE
        [0.0, 0.0],  # j12 vel
    ]

    task_types: dict | None = {
        "reset_base": "root",
        "reset_robot_joints": "joint",
    }

    subterm_dims: dict | None = {
        "reset_base": (6, 2),
        "reset_robot_joints": (12, 2),
    }

    tasks: dict | None = {
        "event": {
            # term
            "reset": {
                # subterm
                "reset_base": {
                    # param
                    "pose": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                    ],
                    # param
                    "velocity": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                    ],
                },
                # subterm
                "reset_robot_joints": [
                    # param
                    "position",
                    "velocity",
                ],
            }
        }
    }

    sample_filter: list[tuple[tuple[float, float], ...]] | None = [
        ((-0.75, 0.5), (0, box_height + 0.6)),
        ((-3.5, -1.2), (height_offset, box_height + height_offset)),
    ]

    # Goals on top of the box must stay valid, so the box-footprint exclusion's z-max is lowered.
    goal_sample_filter: list[tuple[tuple[float, float], ...]] | None = [
        ((-0.75, 0.5), (0, box_height + 0.5)),
        ((-3.5, -0.8), (max(height_offset, 0.65), box_height + height_offset)),
    ]

    filter_dimensions: tuple | None = (0, 4)
    optimal_trajectory_file: str | None = (
        "/workspace/isaaclab/informed-exploration/data/parkour/trajectories_parkour_climb_box_frontal_short.pkl"
    )

    # Nominal task for mask_to_subspace; z is absolute in the env frame, so it holds the standing height.
    default_task: list | None = [
        0.0,
        0.0,  # x, vx
        0.0,
        0.0,  # y, vy
        height_offset,
        0.0,  # z, vz
        0.0,
        0.0,  # roll, v roll
        0.0,
        0.0,  # pitch, v pitch
        0.0,
        0.0,  # yaw, v yaw
        # joint positions and velocities
        0.0,
        0.0,  # j1 LF_HAA
        0.0,
        0.0,  # j2 LH_HAA
        0.0,
        0.0,  # j3 RF_HAA
        0.0,
        0.0,  # j4 RH_HAA
        0.0,
        0.0,  # j5 LF_HFE
        0.0,
        0.0,  # j6 LH_HFE
        0.0,
        0.0,  # j7 RF_HFE
        0.0,
        0.0,  # j8 RH_HFE
        0.0,
        0.0,  # j9 LF_KFE
        0.0,
        0.0,  # j10 LH_KFE
        0.0,
        0.0,  # j11 RF_KFE
        0.0,
        0.0,  # j12 RH_KFE
    ]

    def update_box_height(self, new_height: float) -> None:
        self.box_height = new_height
        self.bounds[4][1] = height_offset + new_height
        self.sample_filter = [
            ((-0.75, 0.5), (0, new_height + 0.6)),
            ((-3.5, -1.2), (max(height_offset, 0.7), new_height + height_offset)),
        ]
        self.goal_sample_filter = [
            ((-0.75, 0.5), (0, new_height + 0.5)),
            ((-3.5, -0.8), (max(height_offset, 0.65), new_height + height_offset)),
        ]
