#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name="eval_local"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Please provide a model name and a task name."
    echo "Usage: $0 <model_name> <task_name>"
    exit 1
fi

export HF_HOME=$SCRATCHDIR/.cache/huggingface
export XDG_CACHE_HOME=$SCRATCHDIR/.cache

# to be filled in
export UV_CACHE_DIR=$SCRATCHDIR/.cache/uv
export WANDB_API_KEY=""
export HF_TOKEN=""
VENV_DIR="$SCRATCHDIR/eval_venv_$(uuidgen)"
PROJECT_DIR="$HOME/projects/treason/eval/lm-evaluation-harness"
RESULTS_DIR="$HOME/projects/treason/results"

# create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/python" ]; then
    echo " Virtualenv not found. Creating and activating a new one with uv"
    uv venv "$VENV_DIR" --clear
    source "$VENV_DIR/bin/activate"

    # install the project dependecies 
    cd "$PROJECT_DIR"
    uv pip install -e .[math]
    uv pip install "lm_eval[vllm]"
    uv pip install ray

    # ONE clean resolution: Pin vLLM, Pin Torch, and explicitly require the missing NVIDIA library
    uv pip install --reinstall-package torch vllm==0.19.1 "torch==2.10.0" torchvision torchaudio nvidia-cusparselt-cu12 nvidia-cuda-runtime-cu12 antlr4-python3-runtime==4.11.1 --extra-index-url https://download.pytorch.org/whl/cu126
else
    source "$VENV_DIR/bin/activate"
    echo "Virtualenv at $VENV_DIR activated"
fi

cd "$PROJECT_DIR"

# Add CUDA 12 runtime and Torch libs to LD_LIBRARY_PATH so vllm can find it
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV_DIR/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

lm_eval --model vllm \
    --model_args pretrained=$SCRATCHDIR/models/$1,tensor_parallel_size=1 \
    --tasks $2 \
    --batch_size auto \
    --output_path $RESULTS_DIR \
    --log_samples \
    --num_fewshot 0 \
    --gen_kwargs "max_gen_toks=2048,temperature=0.0,do_sample=false"

#remove venv 
rm -rf "$VENV_DIR"
echo  "Removed venv at $VENV_DIR"
