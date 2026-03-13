export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
mkdir -p "${HOME}/.triton/autotune"
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    # Single A100-80G setup
    torchrun --nproc_per_node 1 \
            sft.py \
            --base_model ./models/Qwen2.5-Instruct-0.5b \
            --batch_size 256 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output_dir/sft_industrial_qwen25_05b \
            --wandb_project minionerec-sft \
            --wandb_run_name sft-industrial-qwen25-0.5b-seed42 \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
            --freeze_LLM False
done
