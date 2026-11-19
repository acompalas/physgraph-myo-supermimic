#!/usr/bin/env bash
# Fresh-PVC setup for physgraph-myo-supermimic (ucsd-videoprocessing).
# Installs miniconda to /data/miniconda3, then the same physgraph-myo env as
# pod/setup_physgraph_myo_env.sh (verified 2026-09-05), without the
# /data/PhysGraph prerequisite (supermimic does not import it).
# apt packages do not persist: rerun step 1 on every new pod.
# Usage: bash pod/setup_supermimic_pod.sh
set -e
export DEBIAN_FRONTEND=noninteractive

echo "== [0/5] GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== [1/5] System packages =="
apt-get update && apt-get install -y git tmux wget \
    libgl1-mesa-glx libxrender1 libxrandr2 libglib2.0-0 \
    libosmesa6-dev libglfw3 patchelf

echo "== [2/5] Miniconda (skipped if present) =="
if [ ! -x /data/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p /data/miniconda3
fi
source /data/miniconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "== [3/5] physgraph-myo env (Python 3.11) =="
if ! conda env list | grep -q physgraph-myo; then
  conda create -y -n physgraph-myo python=3.11
fi
conda activate physgraph-myo

echo "== [4/5] Python packages (same pins as setup_physgraph_myo_env.sh) =="
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install mujoco==3.12.0
pip install pytorch_kinematics rl_games omegaconf dm-tree gym hydra-core wandb h5py
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
pip install mujoco-warp
pip install numpy==2.4.6

echo "== [5/5] Verify =="
python -c "
import mujoco, mujoco_warp, torch, numpy
print('mujoco', mujoco.__version__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| numpy', numpy.__version__)
print(torch.cuda.get_device_name(0))
print('ALL CORE IMPORTS OK')
"
echo "== Done. python: /data/miniconda3/envs/physgraph-myo/bin/python =="
