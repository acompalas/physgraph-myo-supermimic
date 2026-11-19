#!/usr/bin/env bash
# Fresh-PVC setup for physgraph-myo-supermimic (ucsd-videoprocessing).
# Installs miniconda to /data/miniconda3, then the same physgraph-myo env as
# pod/setup_physgraph_myo_env.sh (verified 2026-09-05), without the
# /data/PhysGraph prerequisite (supermimic does not import it).
# apt packages do not persist: rerun step 1 on every new pod.
# Usage: bash pod/setup_supermimic_pod.sh
#
# Before/after this script (one-time per PVC):
#   git clone --recurse-submodules https://github.com/acompalas/physgraph-myo-supermimic.git /data/physgraph-myo-supermimic
#     (third_party/myo_sim holds the MyoHand meshes; plain clone leaves it empty)
#   copy PhysGraph's physgraph_envs/lib/envs/dexhands/base.py to /data/PhysGraph/<same path>
#     (scripts/retargeting/myohand_def.py inherits DexHand from it)
#   kubectl cp the untracked demos (real files, not symlinks) and sm_data/*.pt, check md5
#   python scripts/sm_check_imports.py  -> expect only cv2 + pytorch_lightning FAILs (unused)
# git/tmux: apt can be unreachable; use conda create -y -p /data/tools -c conda-forge git tmux (persists on the PVC)
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
TORCH_CUDA=cu126  # = Run O; covers sm_50..sm_90 (sm_89 via sm_86); cu130 starts at sm_75
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/$TORCH_CUDA
pip install mujoco==3.12.0
pip install pytorch_kinematics==0.10.0 rl_games==1.6.5 omegaconf dm-tree gym hydra-core wandb h5py
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
pip install warp-lang==1.17.0
pip install --no-deps mujoco-warp==3.12.0  # = Run O; 3.13.0 fails on a cold warp cache (ccd_kernel KeyError), 4.x breaks mujoco 3.12.0
pip install numpy==2.4.6

echo "== [5/5] Verify =="
python -c "
import mujoco, mujoco_warp, torch, numpy
print('mujoco', mujoco.__version__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| numpy', numpy.__version__)
print(torch.cuda.get_device_name(0))
print('ALL CORE IMPORTS OK')
"
echo "== Done. python: /data/miniconda3/envs/physgraph-myo/bin/python =="
