#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Flower + nnU-Net v2 Environment Bootstrap (no ~/.bashrc edits)
# - Auto-install Miniconda to ~/miniconda3 if conda is missing
# - Exports nnU-Net paths ONLY for this session
# - Enforces DatasetXXX_Name (e.g., Dataset009_Spleen)
# =============================================================================
# Usage:
#   bash   setup_flwr_nnunet.sh [--name flower-nnunet] [--cuda cu126|cpu] \
#                               [--dataset Task09_Spleen] [--no-preprocess]
#   source setup_flwr_nnunet.sh  # keep exports in your current shell
# =============================================================================

ENV_NAME="flower-nnunet"
CUDA_FLAVOR="cu126"       # cu126 | cpu
DATASET="Task09_Spleen"   # e.g., Task03_Liver
RUN_PREPROCESS=1
DEMO_DIR="flwr-nnunet-demo"

die() { echo "ERROR: $*" >&2; exit 1; }

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
					          -h|--help) sed -n '1,200p' "$0"; exit 0;;
						      *) die "Unknown arg: $1";;
						        esac
						done

						# -----------------------------
						# Basic tools
						# -----------------------------
						command -v wget >/dev/null 2>&1 || die "wget not found (install wget and retry)."
						command -v tar  >/dev/null 2>&1 || die "tar not found (install tar and retry)."

						# -----------------------------
						# Conda (auto-install Miniconda if missing)
						# -----------------------------
						if ! command -v conda >/dev/null 2>&1; then
							  echo ">> Conda not found. Installing Miniconda to ~/miniconda3 ..."
							    mkdir -p "$HOME/miniconda3"
							      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh"
							        bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
								  rm "$HOME/miniconda3/miniconda.sh"
								    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)" || die "Failed to initialize conda."
							    else
								      eval "$(conda shell.bash hook)" || die "Failed to initialize conda."
						fi

						# -----------------------------
						# CUDA / Torch channel
						# -----------------------------
						case "$CUDA_FLAVOR" in
							  cu126) TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/cu126";;
							    cpu)   TORCH_INDEX_URL="--index-url https://download.pytorch.org/whl/cpu";;
							      *) die "--cuda must be cu126 or cpu";;
						      esac

						      if [[ "$CUDA_FLAVOR" == "cu126" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
							        echo ">> WARNING: cu126 selected but no NVIDIA GPU detected via nvidia-smi. Install will succeed, but GPU won't be usable."
						      fi

						      # -----------------------------
						      # Workspace & nnU-Net paths (exports only)
						      # -----------------------------
						      mkdir -p "$DEMO_DIR"
						      cd "$DEMO_DIR"

						      mkdir -p nnUNet_raw nnUNet_preprocessed nnUNet_results
						      export nnUNet_raw="${nnUNet_raw:-$(pwd)/nnUNet_raw}"
						      export nnUNet_preprocessed="${nnUNet_preprocessed:-$(pwd)/nnUNet_preprocessed}"
						      export nnUNet_results="${nnUNet_results:-$(pwd)/nnUNet_results}"

						      echo ">> nnU-Net paths (exported for this session):"
						      echo "   nnUNet_raw=${nnUNet_raw}"
						      echo "   nnUNet_preprocessed=${nnUNet_preprocessed}"
						      echo "   nnUNet_results=${nnUNet_results}"

						      # -----------------------------
						      # Conda env create/activate  (-y to accept TOS)
						      # -----------------------------
						      CONDA_CREATE_BIN="conda"; command -v mamba >/dev/null 2>&1 && CONDA_CREATE_BIN="mamba"
						      if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
							        echo ">> Conda env '$ENV_NAME' already exists."
							else
								  echo ">> Creating conda env: $ENV_NAME (python=3.10)"
								     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
								     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
								     "$CONDA_CREATE_BIN" create -y -n "$ENV_NAME" python=3.10
						      fi
						      conda activate "$ENV_NAME"
		

						      # -----------------------------
						      # Python deps
						      # -----------------------------
						      python -m pip install --upgrade pip
						      python -m pip install torch torchvision $TORCH_INDEX_URL
						      python -m pip install "flwr[simulation]" wandb nnunetv2 SimpleITK nibabel scikit-learn tqdm psutil pyyaml

						      # Quick sanity: nnUNetv2 CLIs present?
						      command -v nnUNetv2_convert_MSD_dataset >/dev/null 2>&1 || die "nnUNetv2 CLI not found after install."
						      command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1 || die "nnUNetv2 CLI not found after install."

						      # Optional reference repo
						      [[ -d nnUNet ]] || git clone https://github.com/MIC-DKFZ/nnUNet.git

						      # -----------------------------
						      # Dataset fetch & convert
						      # -----------------------------
						      [[ -n "${MSD_LINKS[$DATASET]:-}" ]] || die "Unknown dataset ${DATASET}"

						      mkdir -p dataset && cd dataset
						      ARCHIVE_URL="${MSD_LINKS[$DATASET]}"
						      ARCHIVE_NAME="$(basename "$ARCHIVE_URL")"

						      # Extract only if target folder not present
						      if [[ ! -d "$DATASET" ]]; then
							        [[ -f "$ARCHIVE_NAME" ]] || wget -c "$ARCHIVE_URL"
								  echo ">> Extracting $ARCHIVE_NAME ..."
								    tar -xf "$ARCHIVE_NAME"
						      fi

						      # Convert MSD -> nnU-Net v2 (writes into $nnUNet_raw)
						      nnUNetv2_convert_MSD_dataset -i "./$DATASET"

						      cd ..

						      # -----------------------------
						      # Enforce DatasetXXX_Name naming & verify
						      # -----------------------------
						      TASK_NUM="$(echo "$DATASET" | sed -E 's/Task0*([0-9]+).*/\1/' | awk '{printf "%03d",$0}')"
						      TASK_SUFFIX="$(echo "$DATASET" | sed -E 's/Task0*[0-9]+_//')"
						      EXPECTED_DIR="${nnUNet_raw}/Dataset${TASK_NUM}_${TASK_SUFFIX}"
						      ALT_TASK_DIR="${nnUNet_raw}/Task${TASK_NUM}_${TASK_SUFFIX}"

						      if [[ ! -d "$EXPECTED_DIR" && -d "$ALT_TASK_DIR" ]]; then
							        echo ">> Renaming ${ALT_TASK_DIR} -> ${EXPECTED_DIR}"
								  mv "$ALT_TASK_DIR" "$EXPECTED_DIR"
						      fi

						      [[ -d "$EXPECTED_DIR" ]] || die "Expected dataset folder not found: $EXPECTED_DIR"
						      [[ -f "$EXPECTED_DIR/dataset.json" ]] || die "dataset.json missing in ${EXPECTED_DIR}"

						      # -----------------------------
						      # Plan & Preprocess
						      # -----------------------------
						      if [[ "$RUN_PREPROCESS" -eq 1 ]]; then
							        echo ">> Running plan & preprocess for -d ${TASK_NUM}"
								  nnUNetv2_plan_and_preprocess -d "${TASK_NUM}" --verify_dataset_integrity
							  else
								    echo ">> Skipping plan & preprocess."
						      fi

						      echo
						      echo "============================================================================="
						      echo "Setup complete."
						      echo "Activate env with: conda activate $ENV_NAME"
						      echo "Dataset prepared at: ${EXPECTED_DIR}"
						      echo "Note: nnU-Net paths were only exported for this session."
						      echo "============================================================================="
