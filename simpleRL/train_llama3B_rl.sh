#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --partition=workq
#SBATCH --job-name="simple-rl"
#SBATCH --time=4:00:00

set -e

export HF_HOME=$SCRATCHDIR/.cache/huggingface
export XDG_CACHE_HOME=$SCRATCHDIR/.cache
export UV_CACHE_DIR=$SCRATCHDIR/.cache/uv
export SYMPY_USE_CACHE=no
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export WANDB_CACHE_DIR="$SCRATCHDIR/.cache/wandb"
export WANDB_DIR="$SCRATCHDIR"


# fill in / adjust 
export WANDB_API_KEY=""
export HF_TOKEN=""
SHARED_VENV_DIR="$SCRATCHDIR/simpleRL-reason"
VENV_DIR="$LOCALDIR/simpleRL-reason"
# location of the simpleRL folder
PROJECT_DIR="$HOME/projects/openai_treason/simpleRL"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# hpc specific environment setup

# Prefer a node-local venv on $LOCALDIR (fast NVMe, private to this job). Running the
# venv off shared $SCRATCHDIR triggers bus errors: multiple jobs write the same path
# concurrently, and mmap'd .so files on the parallel FS raise SIGBUS.
if [ -x "$VENV_DIR/bin/python" ]; then
    echo "Node-local virtualenv already present at $VENV_DIR"
elif [ -x "$SHARED_VENV_DIR/bin/python" ]; then
    echo "Found shared virtualenv at $SHARED_VENV_DIR; copying to node-local $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    rm -rf "$VENV_DIR"
    cp -a "$SHARED_VENV_DIR" "$VENV_DIR"
    # The copy still has the shared path baked into its activate scripts, entry-point
    # shebangs and pyvenv.cfg; rewrite them to point at the node-local location.
    grep -rIlZ "$SHARED_VENV_DIR" "$VENV_DIR/bin" 2>/dev/null \
        | xargs -0 -r sed -i "s|$SHARED_VENV_DIR|$VENV_DIR|g"
    [ -f "$VENV_DIR/pyvenv.cfg" ] && sed -i "s|$SHARED_VENV_DIR|$VENV_DIR|g" "$VENV_DIR/pyvenv.cfg"
else
    echo "No virtualenv found. Creating a new node-local one with uv"
    uv venv "$VENV_DIR" --clear --python 3.10
fi
source "$VENV_DIR/bin/activate"

# Load modules required for building flash-attn and xformers from source
# Note: You may need to adjust these module names based on 'module avail' on Isambard
module load gcc-native/12.3 || echo "Warning: gcc-native module not loaded"
module load cuda/12.6 || echo "Warning: cuda module not loaded"
export CC=gcc
export CXX=g++
# Ensure CUDA_HOME is set for flash-attn build
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAX_JOBS=8 # Limit parallel build jobs to prevent OOM

# install the project dependecies 
cd "$PROJECT_DIR"
uv pip install "setuptools<70.0.0" wheel packaging ninja cmake
 
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# Build torchvision from GitHub, but set BUILD_VERSION=0.19.0 so it installs precisely as "0.19.0"
# This prevents uv from rejecting it as "0.19.0a0+git" when installing vllm
BUILD_VERSION=0.19.0 uv pip install "git+https://github.com/pytorch/vision.git@v0.19.0" --no-build-isolation --reinstall-package torchvision

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH! xformers will build as CPU-only. Please fix the CUDA module load."
    exit 1
fi

export FORCE_CUDA="1"
export TORCH_CUDA_ARCH_LIST="9.0" # Grace Hopper architecture

echo "Checking PyTorch CUDA status before compiling xformers..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

uv pip install flash-attn --no-build-isolation
# Compile vLLM's specific fork of flash-attention from source since PyPI lacks ARM64 wheels and source tarballs for it
echo "Compiling vllm-flash-attn 2.6.1... This will take 10-15 minutes."
uv pip install git+https://github.com/vllm-project/flash-attention.git@v2.6.1 --no-build-isolation

# Reinstall xformers to overwrite the cached CPU-only wheel with a proper CUDA build
if ! uv pip show xformers | grep -q "0.0.27.post2"; then
    echo "Compiling xformers... This will take a few minutes. Logging to xformers_build.log"
    # Inject the missing <cuda/atomic> header directly into the compiler to fix the CUDA 12.6 error elegantly
    NVCC_FLAGS="-include cuda/atomic" uv pip install xformers==0.0.27.post2 --no-build-isolation --reinstall-package xformers --no-cache --no-binary xformers > xformers_build.log 2>&1 || {
        echo "ERROR: xformers compilation failed! Here are the actual error lines from the compiler:"
        grep -A 10 -B 2 -i "error:" xformers_build.log | tail -n 40
        exit 1
    }
else
    echo "xformers already installed with CUDA support, skipping compilation."
fi

# Explicitly provide the git repo to uv so it doesn't fail resolution searching the index for missing aarch64 wheels
BUILD_VERSION=0.19.0 uv pip install vllm==0.5.4 torch==2.4.0 "torchvision@git+https://github.com/pytorch/vision.git@v0.19.0" "triton @ git+https://github.com/triton-lang/triton.git@v3.0.0#subdirectory=python" --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu124

uv pip install git+https://github.com/ozeliger/pyairports.git pycountry
uv pip install wandb
uv pip install "ray[default]==2.10.0"

BUILD_VERSION=0.19.0 uv pip install -e "$PROJECT_DIR" torch==2.4.0 "torchvision@git+https://github.com/pytorch/vision.git@v0.19.0" "triton @ git+https://github.com/triton-lang/triton.git@v3.0.0#subdirectory=python" --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install "click<8.1.8"



cd "$PROJECT_DIR"   

mkdir -p $SCRATCHDIR/logs
mkdir -p $SCRATCHDIR/checkpoints
mkdir -p $SCRATCHDIR/models

DATA_DIR="$SCRATCHDIR/simplelr_abel_level1to4"
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    wget -O "$DATA_DIR/train.parquet" https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_abel_level1to4/train.parquet
fi

if [ ! -f "$DATA_DIR/test.parquet" ]; then
    wget -O "$DATA_DIR/test.parquet" https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_abel_level1to4/test.parquet
fi

# Catch termination signals (like scancel) to gracefully stop Ray and free GPUs
trap "echo 'Caught termination signal, stopping Ray...'; ray stop; exit 0" EXIT SIGTERM SIGINT

ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --object-store-memory 20000000000 --temp-dir=$SCRATCHDIR/ray_tmp
sleep 5 # wait for the ray dashboard to boot up
export HEAD_IP=127.0.0.1
export HEAD_PORT=6379

# use the right GPU connection
export NCCL_P2P_DISABLE=1

# Checkpointing writes ~49GB to Lustre and every rank barriers on the slowest
# writer; with many jobs sharing the FS this exceeds NCCL's 10min default and the
# watchdog kills the job. Allow 2h for collectives.
export VERL_NCCL_TIMEOUT=7200

# Prevent thread-induced memory bloat on many-core nodes
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Cap glibc malloc arenas and the Rust tokenizers (Rayon) threadpool: on this
# ~200+ core node an uncapped process reserves up to 8*ncores 64MB arenas, which
# exhausts host RAM / mmap space and triggers the driver OOM during reward scoring.
export MALLOC_ARENA_MAX=2
export RAYON_NUM_THREADS=1

bash "$PROJECT_DIR/train_grpo_math_tune_ray.sh" --model_name Llama-3.2-3B --dataset_name simplelr_abel_level1to4 --max_response_length 2048  --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0.0001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.6 --rollout_tp 1 --save_freq 10 --micro_rollout_batch_size 128 --ppo_micro_batch_size 2
