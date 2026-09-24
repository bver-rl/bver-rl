# 1. IsaacLab fork (pins the rsl_rl fork through isaaclab_rl/setup.py)
git clone -b bver-v1 https://github.com/bver-anon/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install rsl_rl && cd ..

# 2. This repo
git clone https://github.com/bver-anon/bver-rl.git
cd bver-rl && python -m pip install -e source/informed_exploration

# 3. Sanity check
python scripts/list_envs.py