if [ ! -f /root/.cache/concatenated_all.jsonl ]; then
    mkdir -p /root/.cache
    cat "/mnt/src/data/r2egym_splits/r2egym_d1p_claude_v2_messages_train_1k.jsonl" \
        "/mnt/src/data/r2egym_splits_sep20/r2egym_d4_claude_v2_messages_train.jsonl" \
        "/mnt/src/data/r2egym_splits/r2egym_d2p_claude_v2_messages_train_1k.jsonl" \
        "/mnt/src/data/r2egym_splits_sep21/r2egym_d4_claude_v2_messages_train.jsonl" \
        "/mnt/src/data/r2egym_splits/r2egym_d1+d2_claude_v2_messages_train.jsonl" \
        > /root/.cache/concatenated_all.jsonl
    echo "Created /root/.cache/concatenated_all.jsonl"
else
    echo "/root/.cache/concatenated_all.jsonl already exists, skipping concatenation"
fi

# setting to TRANSFORMERS_CACHE
export MODELSCOPE_CACHE=/mnt/cache/hub/
# 8 * 57GiB, 2.95s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --load_safetensors true \
    --dataset "/root/.cache/concatenated_all.jsonl" \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 8 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size ${TOTAL_GPUS:-8} \
    --packing false \
    --moe_router_dtype fp32 \
    --recompute_granularity full \
    --recompute_method block \
    --recompute_num_layers 8 \
    --max_epochs 10 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save $save_dir \
    --eval_interval 1000 \
    --save_interval 1000 \
    --max_length 32768 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --overlap_param_gather true \
    --overlap_grad_reduce true \
    --no_initialization false \
    --bf16 true \
    --truncation_strategy "right" \
    --wandb_project $WANDB_PROJECT \
    --wandb_exp_name $WANDB_RUN_NAME \
    --exp_avg_dtype fp16 \
    --exp_avg_sq_dtype fp16 \
    --use_precision_aware_optimizer true \


# --cached_dataset "" \
# --load_from_cache_file true \

# --cached_dataset "/root/.cache/concatenated_all_cached" \

# --load /mnt/cache/hub/models--Qwen--Qwen3-30B-A3B \
#  --recompute_num_layers 1 \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# megatron sft \
#     --model Qwen/Qwen3-30B-A3B \
#     --load_safetensors true \
#     --dataset "/root/.cache/concatenated_all.jsonl" \
#     --split_dataset_ratio 0.01 \
#     --tensor_model_parallel_size 4 \
#     --pipeline_model_parallel_size 2 \
#     --context_parallel_size 2 \
#     --expert_model_parallel_size 2 \
#     --moe_permute_fusion true \
#     --moe_grouped_gemm true \
#     --moe_shared_expert_overlap true \
#     --moe_aux_loss_coeff 1e-3 \
#     --micro_batch_size 1 \
#     --global_batch_size ${TOTAL_GPUS:-8} \
#     --packing false \
#     --moe_router_dtype fp32 \
#     --recompute_granularity full \
#     --recompute_method block \
#     --recompute_num_layers 8 \
#     --max_epochs 1 \
#     --finetune true \
#     --cross_entropy_loss_fusion true \
#     --lr 1e-5 \
#     --lr_warmup_fraction 0.05 \
#     --min_lr 1e-6 \
#     --save $save_dir \
#     --eval_interval 200 \
#     --save_interval 200 \
#     --max_length 32768 \
#     --num_workers 8 \
#     --dataset_num_proc 8 \
#     --no_save_optim true \
#     --no_save_rng true \
#     --sequence_parallel true \
#     --attention_backend flash \
#     --overlap_param_gather true \
#     --overlap_grad_reduce true \
#     --no_initialization false \
#     --bf16 true \
#     --truncation_strategy "right" \
#     --wandb_project $WANDB_PROJECT \
#     --wandb_exp_name $WANDB_RUN_NAME \
#     --optimizer_cpu_offload true \
#     --use_precision_aware_optimizer true \
#     --optimizer_offload_fraction 1 \
#     --exp_avg_dtype fp16 \
#     --exp_avg_sq_dtype fp16 \
#     --use_cpu_initialization true

# # --cached_dataset "" \
# # --load_from_cache_file true \

# # --cached_dataset "/root/.cache/concatenated_all_cached" \

# # --load /mnt/cache/hub/models--Qwen--Qwen3-30B-A3B \
# #  --recompute_num_layers 1 \
