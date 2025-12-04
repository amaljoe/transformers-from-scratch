export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

cd ~/workspace/transformers-from-scratch
source .venv/bin/activate

# python3 -c "import torch; import transformers; import datasets"
