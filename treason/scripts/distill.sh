#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name="distill"
export HF_HOME=$SCRATCHDIR/.cache/huggingface
export XDG_CACHE_HOME=$SCRATCHDIR/.cache
export UV_CACHE_DIR=$SCRATCHDIR/.cache/uv
export WANDB_CACHE_DIR="$SCRATCHDIR/.cache/wandb"
export WANDB_DIR="$SCRATCHDIR"

# fill in values
export WANDB_API_KEY=""
export HF_TOKEN=""
VENV_DIR="$SCRATCHDIR/treason_distill_venv"
PROJECT_DIR="$HOME/projects/openai_treason"

# variables 
DATASET=""
OUTPUT_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --output_name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$DATASET" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Error: Please provide --dataset and --output_name."
    echo "Usage: $0 --dataset <dataset_path> --output_name <output_dir>"
    exit 1
fi

# hpc specific environment setup

# create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/python" ]; then
    echo " Virtualenv not found. Creating and activating a new one with uv"
    uv venv "$VENV_DIR" --clear
    source "$VENV_DIR/bin/activate"
    echo "Virtualenv at $VENV_DIR created and activated"

    # install configured dependencies
    uv sync --active 

    # ONE clean resolution: Pin vLLM, Pin Torch, and explicitly require the missing NVIDIA library
    uv pip install "torch==2.5.1" torchvision torchaudio nvidia-cusparselt-cu12 --extra-index-url https://download.pytorch.org/whl/cu124 

else
    source "$VENV_DIR/bin/activate" 
    echo "Virtualenv at $VENV_DIR activated"
    
    # # install configured dependencies
    # uv sync --active 

    # # ONE clean resolution: Pin vLLM, Pin Torch, and explicitly require the missing NVIDIA library
    # uv pip install "torch==2.5.1" torchvision torchaudio nvidia-cusparselt-cu12 --extra-index-url https://download.pytorch.org/whl/cu124 
fi
cd $PROJECT_DIR
echo "$PROJECT_DIR/$DATASET"
python treason/src/distillation/distill_expanded_traces.py \
    --model_name="meta-llama/Llama-3.2-3B" \
    --block_size=32768 \
    --training_dataset_path="$PROJECT_DIR/$DATASET" \
    --output_dir="$SCRATCHDIR/models/$OUTPUT_NAME" \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --warmup_ratio=0.1 \
    --lr_scheduler_type="cosine" \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --bf16=True \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --run_name="$OUTPUT_NAME" \
    --report_to="wandb"
