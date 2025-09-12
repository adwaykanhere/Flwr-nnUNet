#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Flower + nnU-Net v2 Environment Bootstrap
# - Creates nnU-Net paths
# - Sets env vars in ~/.bashrc (idempotent)
# - Creates conda env and installs PyTorch, Flower, nnUNetv2, and deps
# - Downloads a selected MSD dataset and converts to nnU-Net v2 format
# - Runs planning & preprocessing
#
# Usage:
#   bash setup_flwr_nnunet.sh [--name flower-nnunet] [--cuda cu126|cpu] \
#     [--dataset Task09_Spleen] [--no-preprocess]
#
# Examples:
#   bash setup_flwr_nnunet.sh
#   bash setup_flwr_nnunet.sh --dataset Task03_Liver --cuda cu126
#   bash setup_flwr_nnunet.sh --cuda cpu --no-preprocess
#
# Requires:
#   - Conda (mamba/conda) pre-installed and on PATH
#   - wget, tar
# =============================================================================

# -----------------------------
# Defaults
# -----------------------------
ENV_NAME="flower-nnunet"
CUDA_FLAVOR="cu126"       # "cu126" or "cpu"
DATASET="Task09_Spleen"   # change via --dataset Task0X_Name
RUN_PREPROCESS=1
DEMO_DIR="flwr-nnunet-demo"

# MSD dataset links
declare -A MSD_LINKS=(
  ["Task01_BrainTumour"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
  ["Task02_Heart"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"
  ["Task03_Liver"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
  ["Task04_Hippocampus"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
  ["Task05_Prostate"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar"
  ["Task06_Lung"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar"
  ["Task07_Pancreas"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
  ["Task08_HepaticVessel"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar"
  ["Task09_Spleen"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
  ["Task10_Colon"]="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar"
)

# -----------------------------
# Args
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) ENV_NAME="$2"; shift 2;;
    --cuda) CUDA_FLAVOR="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --no-preprocess) RUN_PREPROCESS=0; shift 1;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda not found on PATH. Please install Miniconda/Anaconda and retry."
  exit 1
fi
if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget not found. Install wget and retry."
  exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "ERROR: tar not found. Install tar and retry."
  exit 1
fi

# -----------------------------
# CUDA / Torch channel
# -----------------------------
TORCH_INDEX_URL=""
case "$CUDA_FLAVOR" in
  cu126) TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/cu126";;
  cpu)   TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/cpu";;
  *)
    echo "ERROR: --cuda must be cu126 or cpu"; exit 1;;
esac

# -----------------------------
# Prepare workspace & nnU-Net paths
# -----------------------------
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

mkdir -p nnUNet_raw nnUNet_preprocessed nnUNet_results

# Append to ~/.bashrc idempotently
insert_if_missing() {
  local line="$1"
  local file="$2"
  grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

BASHRC="$HOME/.bashrc"
insert_if_missing "export nnUNet_raw='$(pwd)/nnUNet_raw'" "$BASHRC"
insert_if_missing "export nnUNet_preprocessed='$(pwd)/nnUNet_preprocessed'" "$BASHRC"
insert_if_missing "export nnUNet_results='$(pwd)/nnUNet_results'" "$BASHRC"

echo ">> nnU-Net paths ensured and exported to ~/.bashrc"
echo "   nnUNet_raw=$(pwd)/nnUNet_raw"
echo "   nnUNet_preprocessed=$(pwd)/nnUNet_preprocessed"
echo "   nnUNet_results=$(pwd)/nnUNet_results"

# -----------------------------
# Conda env create / update
# -----------------------------
# Use mamba if available for speed
CONDA_CREATE_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_CREATE_BIN="mamba"
fi

# Create env if missing
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> Conda env '$ENV_NAME' already exists. Skipping creation."
else
  echo ">> Creating conda env: $ENV_NAME (python=3.10)"
  "$CONDA_CREATE_BIN" create -n "$ENV_NAME" python=3.10 -y
fi

# Activate env in script
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# -----------------------------
# Install Python deps
# -----------------------------
echo ">> Installing PyTorch + torchvision + deps ($CUDA_FLAVOR)"
python -m pip install --upgrade pip

# Core stack
python -m pip install torch torchvision $TORCH_INDEX_URL

# Scientific & medical imaging stack + Flower + nnUNetv2
python -m pip install \
  "flwr[simulation]" \
  wandb \
  nnunetv2 \
  SimpleITK \
  nibabel \
  scikit-learn

# Helpful utils
python -m pip install tqdm psutil pyyaml

# -----------------------------
# Clone nnU-Net (reference, optional)
# -----------------------------
if [[ ! -d "nnUNet" ]]; then
  echo ">> Cloning MIC-DKFZ/nnUNet (reference repo)"
  git clone https://github.com/MIC-DKFZ/nnUNet.git
else
  echo ">> nnUNet repo already present."
fi

# -----------------------------
# Dataset fetch & convert
# -----------------------------
if [[ -z "${MSD_LINKS[$DATASET]:-}" ]]; then
  echo "ERROR: Unknown dataset '$DATASET'."
  echo "Valid options:"
  for k in "${!MSD_LINKS[@]}"; do echo "  - $k"; done
  exit 1
fi

mkdir -p dataset
cd dataset

ARCHIVE_URL="${MSD_LINKS[$DATASET]}"
ARCHIVE_NAME="$(basename "$ARCHIVE_URL")"

if [[ ! -f "$ARCHIVE_NAME" && ! -d "$DATASET" ]]; then
  echo ">> Downloading $DATASET (~large)"
  wget -c "$ARCHIVE_URL"
fi

if [[ -f "$ARCHIVE_NAME" && ! -d "$DATASET" ]]; then
  echo ">> Extracting $ARCHIVE_NAME"
  tar -xf "$ARCHIVE_NAME"
fi

if [[ ! -d "$DATASET" ]]; then
  echo "ERROR: Expected directory '$DATASET' after extract. Aborting."
  exit 1
fi

echo ">> Converting MSD dataset to nnU-Netv2: $DATASET"
nnUNetv2_convert_MSD_dataset -i "./$DATASET"

cd ..

# -----------------------------
# Plan & Preprocess
# -----------------------------
if [[ "$RUN_PREPROCESS" -eq 1 ]]; then
  # Derive numeric ID (e.g., Task09_Spleen -> 009)
  TASK_NUM="$(echo "$DATASET" | sed -E 's/Task0*([0-9]+).*/\1/' | awk '{printf "%03d", $0}')"
  echo ">> Running plan & preprocess for dataset id: ${TASK_NUM}"
  nnUNetv2_plan_and_preprocess -d "$TASK_NUM" --verify_dataset_integrity
else
  echo ">> Skipping plan & preprocess (per --no-preprocess)."
fi

# -----------------------------
# Final notes
# -----------------------------
echo
echo "============================================================================="
echo "Setup complete."
echo "Activate your env with:  conda activate $ENV_NAME"
echo "nnU-Net paths were added to ~/.bashrc (open a new shell or 'source ~/.bashrc')."
echo
echo "Dataset prepared: $DATASET"
echo "Workspace: $(pwd)"
echo "============================================================================="
