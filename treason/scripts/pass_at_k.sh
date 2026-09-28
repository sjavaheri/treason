#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name="reason-dataset"
export HF_HOME=$SCRATCHDIR/.cache/huggingface
export XDG_CACHE_HOME=$SCRATCHDIR/.cache
export UV_CACHE_DIR=$SCRATCHDIR/.cache/uv

# fill in
export HF_TOKEN=""

# Load modules required for building from source on ARM64 / Grace Hopper
module load gcc-native/12.3 || echo "Warning: gcc-native module not loaded"
module load cuda/12.6 || echo "Warning: cuda module not loaded"
export CC=gcc
export CXX=g++
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAX_JOBS=8
export FORCE_CUDA="1"
export TORCH_CUDA_ARCH_LIST="9.0" # Grace Hopper architecture

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH! vLLM will fail to build CUDA kernels. Please fix the CUDA module load."
    exit 1
fi
# tag venv with a uuid
VENV_DIR="$LOCALDIR/treason_pass_at_k_$(uuidgen)"
PROJECT_DIR="$HOME/projects/treason"

# create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Virtualenv not found. Creating and activating a new one with uv"
    uv venv "$VENV_DIR" --clear
    source "$VENV_DIR/bin/activate"
    echo "Virtualenv at $VENV_DIR created and activated"

    # install configured dependencies
    uv sync --active

    cd "$HOME/projects/treason/eval/lm-evaluation-harness"
    uv pip install -e .[math]
    uv pip install ray

    # ONE clean resolution: Pin vLLM, Pin Torch, and explicitly require the missing NVIDIA library
    uv pip install --reinstall-package torch vllm==0.19.1 "torch==2.10.0" torchvision torchaudio nvidia-cusparselt-cu12 nvidia-cuda-runtime-cu12 antlr4-python3-runtime==4.11.1 --extra-index-url https://download.pytorch.org/whl/cu126

else
    source "$VENV_DIR/bin/activate"
    echo "Virtualenv at $VENV_DIR activated"

    # Optional: fast sync if dependencies change in pyproject.toml
    uv sync --active

    cd "$HOME/projects/treason/eval/lm-evaluation-harness"
    uv pip install -e .[math]
    uv pip install ray

    # ONE clean resolution: Pin vLLM, Pin Torch, and explicitly require the missing NVIDIA library
    uv pip install --reinstall-package torch vllm==0.19.1 "torch==2.10.0" torchvision torchaudio nvidia-cusparselt-cu12 nvidia-cuda-runtime-cu12 antlr4-python3-runtime==4.11.1 --extra-index-url https://download.pytorch.org/whl/cu126
fi

echo "Checking PyTorch CUDA status..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('HIP:', torch.version.hip)" || true

cd "$PROJECT_DIR"

# Add CUDA 12 runtime and Torch libs to LD_LIBRARY_PATH so vllm can find it
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV_DIR/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Pass through model name and display name, and any remaining CLI args
MODEL_NAME="$1"
DISPLAY_NAME="$2"
DATASET_NAME="$3"
shift 3

python src/datasets/pass_at_k.py --model_name "$MODEL_NAME" --display_name "$DISPLAY_NAME" --dataset_name "$DATASET_NAME" "$@"
