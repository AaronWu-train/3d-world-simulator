#!/usr/bin/env bash
#SBATCH --job-name=amber-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=train-all-%j.log

# Submit FROM the extracted project root:
#   sbatch --partition=YOUR_GPU_PARTITION scripts/train_all.slurm
# Add --account=YOUR_ACCOUNT if required by your cluster.
# Requires Conda and internet on first run (packages + LPIPS VGG weights).
# Load your cluster's Conda module before sbatch if Conda is not on PATH.
# Resource flags follow https://slurm.schedmd.com/sbatch.html
set -Eeuo pipefail
trap 'echo "ERROR: line $LINENO; see train-all-${SLURM_JOB_ID:-local}.log" >&2' ERR

cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f scripts/train_vae.py && -f environment.yml ]] || {
  echo 'Submit from the extracted project root, or set PROJECT_DIR.' >&2
  exit 1
}

# Optional overrides, e.g.:
# VAE_EPOCHS=100 sbatch --partition=gpu scripts/train_all.slurm
CONDA_ENV="${CONDA_ENV:-3dws}"
AUTO_SETUP="${AUTO_SETUP:-1}"
DATA_ROOT="${DATA_ROOT:-$PWD/data}"
OUT_ROOT="${OUT_ROOT:-$PWD/runs/amber}"
VAE_EPOCHS="${VAE_EPOCHS:-500}"
ITERS="${ITERS:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-4}"
SEED="${SEED:-42}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

# Find Conda without depending on the source machine's absolute paths.
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ ! -x "$CONDA_EXE" ]]; then
  for base in "$HOME/miniforge3" "$HOME/.miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -x "$base/bin/conda" ]]; then
      CONDA_EXE="$base/bin/conda"
      break
    fi
  done
fi
[[ -x "$CONDA_EXE" ]] || { echo 'Install Conda or load the cluster Conda module first.' >&2; exit 1; }
eval "$("$CONDA_EXE" shell.bash hook)"
if ! conda activate "$CONDA_ENV"; then
  [[ "$AUTO_SETUP" == 1 ]] || { echo "Missing Conda environment: $CONDA_ENV" >&2; exit 1; }
  conda env create --name "$CONDA_ENV" --file environment.yml
  conda install --name "$CONDA_ENV" --channel conda-forge --yes cxx-compiler make ninja
  conda activate "$CONDA_ENV"
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torchvision==0.26.0
fi

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
# Keep downloads and JIT artifacts on the destination machine. Do not copy
# compiled extensions between machines or override Slurm's CUDA_VISIBLE_DEVICES.
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
mkdir -p "$OUT_ROOT" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"

python - "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path
import torch
import torchvision, lpips, gsplat, spconv.pytorch

assert torch.cuda.is_available(), 'No CUDA GPU visible in this Slurm job'
assert torch.cuda.is_bf16_supported(), 'Dynamics training requires a GPU with bfloat16 support'
for split in ('train', 'val'):
    root = Path(sys.argv[1]) / 'amber_cloth_estimated' / split
    assert list(root.glob('*.hdf5')), f'No episodes in {root}'
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY

echo '=== 1/3: VAE training ==='
vae_dir="$OUT_ROOT/vae"
resume=()
if [[ -f "$vae_dir/checkpoints/latest.pth" ]]; then
  resume=(--resume "$vae_dir/checkpoints/latest.pth")
fi
python -u scripts/train_vae.py \
  --task amber_cloth_estimated --data-root "$DATA_ROOT" --out-dir "$vae_dir" \
  --n-epochs "$VAE_EPOCHS" --iters "$ITERS" --num-workers "$NUM_WORKERS" \
  --seed "$SEED" "${resume[@]}"

echo '=== 2/3: Export train and val latents ==='
vae_ckpt="$vae_dir/checkpoints/best.pth"
[[ -f "$vae_ckpt" ]] || { echo "Missing checkpoint: $vae_ckpt" >&2; exit 1; }
# Bind latents AND dynamics to this exact VAE checkpoint. Re-submitting uses
# existing completed episodes; a changed VAE gets separate output directories.
vae_hash="$(sha256sum "$vae_ckpt")"
vae_hash="${vae_hash%% *}"
latent_dir="$OUT_ROOT/latents/$vae_hash"
for split in train val; do
  python -u scripts/export_latents.py --vae-ckpt "$vae_ckpt" \
    --packed-dir "$DATA_ROOT/amber_cloth_estimated/$split" \
    --out-dir "$latent_dir/$split"
done

echo '=== 3/3: Dynamics training (500 epochs) ==='
dynamics_dir="$OUT_ROOT/dynamics/$vae_hash"
resume=()
if [[ -f "$dynamics_dir/checkpoints/latest.pth" ]]; then
  resume=(--resume "$dynamics_dir/checkpoints/latest.pth")
fi
python -u scripts/train_dynamics.py \
  --latent-dir "$latent_dir/train" --val-latent-dir "$latent_dir/val" \
  --out-dir "$dynamics_dir" --n-history 8 --object-alpha 1 \
  --ar-schedule 1 2 3 4 5 6 7 8 9 10 \
  --ar-schedule-epochs 50 100 150 200 250 300 350 400 450 500 \
  --batch-size "$DYNAMICS_BATCH_SIZE" --iters "$ITERS" \
  --num-workers "$NUM_WORKERS" --seed "$SEED" "${resume[@]}"

echo "Done. VAE: $vae_dir"
echo "Latents: $latent_dir"
echo "Dynamics: $dynamics_dir"
