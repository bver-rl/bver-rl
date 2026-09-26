# BVER: Bidirectional Voronoi-biased Exploration Curriculum for Reinforcement Learning

![BVER grows intermediate start states outward from the goal (blue) and intermediate goals outward from the initial distribution (green), shown on quadrupedal box climbing, ring-on-peg transfer and a scanned terrain.](docs/cover.jpg)

BVER is an automatic curriculum for sparse-reward, long-horizon goal-conditioned RL. Inspired by bidirectional RRT, it grows start states outward from the goal and goals outward from the initial state distribution. Both sets are biased toward unexplored task space and toward each other, and one shared policy trains on both. No demonstrations or reward shaping are needed.

Project page and interactive demo: https://bver-rl.github.io/

## Installation

```bash
# 1. IsaacLab fork (pins the rsl_rl fork through isaaclab_rl/setup.py)
git clone -b bver-v1 https://github.com/bver-rl/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install rsl_rl && cd ..

# 2. This repo
git clone https://github.com/bver-rl/bver-rl.git
cd bver-rl && python -m pip install -e source/informed_exploration

# 3. Sanity check
python scripts/list_envs.py
```

### Tasks

| Paper | BVER task ID | Experiment file |
|---|---|---|
| U-maze | `Maze-U-BVER-v0` | `experiments/maze/u.yaml` |
| N-maze | `Maze-Serpentine-BVER-v0` | `experiments/maze/serpentine.yaml` |
| Large maze | `Maze-Scatter-BVER-v0` | `experiments/maze/scatter.yaml` |
| Ring-on-peg transfer | `Franka-Hanoi-P12-BVER-v0` | `experiments/hanoi/p12.yaml` |
| Climb box up, 0.4 m / 0.7 m | `Parkour-ClimbSingleBox-BVER-sparse-0p41-v0` / `-0p72-v0` | `experiments/parkour/climb_single_box_sparse_v1.yaml` / `_v1_72.yaml` |
| Climb box down, 0.4 m / 0.7 m | `Parkour-ClimbSingleBoxDown-BVER-sparse-0p41-v0` / `-0p72-v0` | `experiments/parkour/climb_single_box_down_sparse_v1.yaml` / `_v1_72.yaml` |
| Three-route terrain | `Parkour-BridgeDTSG-BVER-v0` | `experiments/bridge/bridge_dtsg_v1.yaml` |
| Freestanding boulder I | `Parkour-Scan-BoulderSmallRocks-BVER-v0` | `experiments/scan/scan_boulder_small_rocks_v1.yaml` |
| Freestanding boulder II | `Parkour-Scan-FreestandingBoulder-BVER-v0` | `experiments/scan/scan_freestanding_boulder_v1.yaml` |
| Rocky ascent | `Parkour-Scan-RockyAscent-BVER-v0` | `experiments/scan/scan_rocky_ascent_v1.yaml` |
| Small stone field | `Parkour-Scan-SmallRocks-BVER-v0` | `experiments/scan/scan_small_rocks_v1.yaml` |

The box heights are 0.41 m and 0.72 m (`0p41`, `0p72`), rounded to 0.4 m and 0.7 m in the paper. `python scripts/list_envs.py` prints every registered task.

### Baselines and ablations

All arms of one task share the environment and differ only in the task ID:

| Arm | Task ID pattern |
|---|---|
| BVER (bidirectional) | `…-BVER-…` |
| Forwards expansion only | `…-BVER-fwd-…` |
| Backwards expansion only | `…-BVER-bwd-…` |
| Vanilla PPO | `…-Baseline-…` |
| Random curriculum | `…-Random-…`, `…-RandomGS-…` |
| Reverse curriculum (RC) | `…-RC-…` |
| RSI | `…-RSI-…` |
| Backplay | `…-Backplay-…` |

The connect ratio is set per direction and is 0.5 by default. The maze experiment files include the connect-ratio ablation as separate runs (`…_connect0`, `…_connect1`); on the command line, override both fields:

```bash
python scripts/rsl_rl/train.py --task Maze-U-BVER-v0 --seed 572857 --headless \
    env.curriculum.initialization.goal_connect_ratio=0.0 \
    env.curriculum.initialization.start_connect_ratio=0.0
```

Some arms read precomputed data from `data/`: the ring-on-peg poses (`data/hanoi/hanoi_p12_poses.yaml`, from `scripts/helpers/hanoi_solve_poses.py`), the feasible-start pools of the random curriculum (`data/parkour/feasible_starts_*.pkl`, from the `…-GenFeasibleStarts-v0` tasks), and the demonstrations of RSI and Backplay (`data/parkour/trajectories_*.pkl`). The `.pkl` files are stored with Git LFS; run `git lfs pull` after cloning.

