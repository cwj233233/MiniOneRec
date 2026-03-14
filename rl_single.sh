#!/bin/bash

set -euo pipefail

export NCCL_IB_DISABLE=1
mkdir -p "${HOME}/.triton/autotune"

# Single A100 startup config.
MODEL_PATH="/root/lanyun-tmp/MiniOneRec/models/Qwen2.5-Instruct-0.5b"
OUTPUT_DIR="./output_dir/rl_industrial_qwen25_05b_single"
WANDB_PROJECT="minionerec-rl"
WANDB_RUN_NAME="rl-industrial-qwen25-0.5b-a100-single"

TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
NUM_GENERATIONS=8

for category in "Industrial_and_Scientific"; do
    train_files=(./data/Amazon/train/"${category}"*.csv)
    eval_files=(./data/Amazon/valid/"${category}"*11.csv)
    info_files=(./data/Amazon/info/"${category}"*.txt)

    if [[ ! -e "${train_files[0]}" || ! -e "${eval_files[0]}" || ! -e "${info_files[0]}" ]]; then
        echo "Missing input files for category: ${category}"
        exit 1
    fi

    train_file="${train_files[0]}"
    eval_file="${eval_files[0]}"
    info_file="${info_files[0]}"

    echo "Using train file: ${train_file}"
    echo "Using eval file: ${eval_file}"
    echo "Using info file: ${info_file}"

    torchrun --nproc_per_node 1 \
        rl.py \
        --model_path "${MODEL_PATH}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --num_train_epochs 2 \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --train_file "${train_file}" \
        --eval_file "${eval_file}" \
        --info_file "${info_file}" \
        --category "${category}" \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ranking \
        --num_generations "${NUM_GENERATIONS}" \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model False \
        --beam_search False \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir "${OUTPUT_DIR}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        --wandb_mode online \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
