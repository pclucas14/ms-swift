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
    --model Qwen/Qwen3-32B \
    --load_safetensors true \
    --dataset "/root/.cache/concatenated_all.jsonl" \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size ${TOTAL_GPUS:-8} \
    --packing false \
    --recompute_granularity full \
    --recompute_method block \
    --recompute_num_layers 8 \
    --max_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 2e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 0 \
    --save $save_dir \
    --save_interval 10000 \
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
    --optimizer_offload_fraction 0.3 \
    --optimizer_cpu_offload true \
