# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sparse Top-K Distillation Trainer for Megatron.

This trainer implements knowledge distillation using pre-computed top-k token 
probabilities from a teacher model. The teacher data is stored sparsely (only
for assistant tokens) in .pt shard files.

Key features:
- No teacher model at training time (uses pre-computed data)
- Sparse storage: only teacher data for assistant tokens
- Renormalized top-k distribution for KL loss
- TP-aware sparse gathering for distributed training
"""
import os
from functools import partial
from glob import glob
from typing import Any, Dict, List, Optional

import torch
from megatron.core import mpu
from megatron.training import get_args, get_timers

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronSparseDistillTrainer(BaseMegatronTrainer):
    """Trainer for sparse top-k distillation.
    
    Uses pre-computed top-k teacher probabilities stored in .pt shard files.
    Computes KL divergence loss between student and renormalized teacher distribution.
    """

    def __init__(self, args, template, **kwargs):
        super().__init__(args, template)
        
        # Distillation parameters
        self.distill_alpha = getattr(args, 'distill_alpha', 1.0)
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)
        self.temperature = getattr(args, 'temperature', 1.0)
        
        # Validate PP=1 using swift args (Megatron args not initialized yet)
        pp_size = getattr(args, 'pipeline_model_parallel_size', 1)
        if pp_size > 1:
            raise ValueError(
                'Sparse distillation does not support Pipeline Parallelism (PP > 1). '
                f'Got PP={pp_size}. Please set --pipeline_model_parallel_size 1'
            )
        
        logger.info(f'SparseDistillTrainer initialized: distill_alpha={self.distill_alpha}, '
                    f'sft_alpha={self.sft_alpha}, temperature={self.temperature}')

    def sparse_topk_kl_div(
        self,
        student_logits: torch.Tensor,
        teacher_top_k_tokens: torch.Tensor,
        teacher_top_k_log_probs: torch.Tensor,
        teacher_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence using sparse top-k teacher distribution.
        
        KL(P_teacher || P_student) = Σ_i P_teacher[i] * (log P_teacher[i] - log P_student[i])
        
        This implementation gathers student logits at teacher's top-k positions,
        then applies log_softmax only over those k logits. This effectively assigns
        a small implicit probability to tokens outside top-k (inspired by NeMo RL's
        zero_outside_topk=False approach).
        
        Args:
            student_logits: Student logits [num_tokens, vocab_size] (vocab may be sharded for TP)
            teacher_top_k_tokens: Top-k token indices [num_tokens, k]
            teacher_top_k_log_probs: Top-k log probabilities [num_tokens, k]
            teacher_mask: Valid position mask [num_tokens] (for padded batches)
            chunk_size: Number of tokens to process per chunk
            
        Returns:
            Tuple of (kl_loss_sum, num_valid_tokens) for proper CP aggregation
        """
        # DO NOT REMOVE BREAKPOINT
        # import pdb; pdb.set_trace()

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_group = mpu.get_tensor_model_parallel_group()
        
        num_tokens = student_logits.shape[0]
        partition_vocab_size = student_logits.shape[-1]
        k = teacher_top_k_tokens.shape[-1]
        
        # Determine valid tokens
        if teacher_mask is not None:
            valid_mask = teacher_mask
            num_valid = valid_mask.sum().float()
        else:
            valid_mask = None
            num_valid = torch.tensor(num_tokens, device=student_logits.device, dtype=torch.float)
        
        if num_valid == 0:
            return student_logits.new_zeros(()), num_valid
        
        # Apply temperature to student logits
        student_logits = student_logits / self.temperature
        
        # Renormalize teacher top-k to form valid probability distribution
        # teacher_probs = softmax(teacher_top_k_log_probs) over the k dimension
        teacher_log_probs_normalized = teacher_top_k_log_probs - torch.logsumexp(
            teacher_top_k_log_probs, dim=-1, keepdim=True
        )
        teacher_probs = torch.exp(teacher_log_probs_normalized)
        
        total_kl = student_logits.new_zeros(())
        total_entropy = student_logits.new_zeros(())
        
        # Process in chunks for memory efficiency
        for start_idx in range(0, num_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, num_tokens)
            
            # Get chunk data
            student_chunk = student_logits[start_idx:end_idx]  # [chunk, partition_vocab]
            teacher_tokens_chunk = teacher_top_k_tokens[start_idx:end_idx]  # [chunk, k]
            teacher_probs_chunk = teacher_probs[start_idx:end_idx]  # [chunk, k]
            teacher_log_probs_chunk = teacher_log_probs_normalized[start_idx:end_idx]  # [chunk, k]
            
            if valid_mask is not None:
                chunk_mask = valid_mask[start_idx:end_idx]
            else:
                chunk_mask = None
            
            # Gather student logits at teacher's top-k positions (TP-aware)
            # Then apply log_softmax only over those k logits
            if tp_size > 1:
                vocab_start = tp_rank * partition_vocab_size
                vocab_end = vocab_start + partition_vocab_size
                
                # Create local indices (relative to this rank's partition)
                local_indices = teacher_tokens_chunk - vocab_start  # [chunk, k]
                
                # Mask for tokens in this rank's range
                in_range = (teacher_tokens_chunk >= vocab_start) & (teacher_tokens_chunk < vocab_end)
                
                # Clamp indices to valid range for gather
                local_indices = local_indices.clamp(min=0, max=partition_vocab_size - 1)
                
                # Gather student logits (not log probs)
                student_logits_at_topk = torch.gather(
                    student_chunk, dim=-1, index=local_indices
                )  # [chunk, k]
                
                # Zero out logits from tokens not in this rank's range
                # Use large negative value for out-of-range to not affect softmax
                student_logits_at_topk = torch.where(
                    in_range,
                    student_logits_at_topk,
                    torch.full_like(student_logits_at_topk, -1e9)
                )
                
                # All-reduce with MAX to get the correct logits from the owning rank
                torch.distributed.all_reduce(
                    student_logits_at_topk, 
                    op=torch.distributed.ReduceOp.MAX, 
                    group=tp_group
                )
            else:
                # No TP: direct gather of logits
                student_logits_at_topk = torch.gather(
                    student_chunk, dim=-1, index=teacher_tokens_chunk
                )  # [chunk, k]
            
            # Apply log_softmax only over the k gathered logits
            # This normalizes the student distribution over the top-k support,
            # effectively assigning small probability to tokens outside top-k
            student_log_probs_at_topk = torch.nn.functional.log_softmax(
                student_logits_at_topk, dim=-1
            )  # [chunk, k]
            student_probs_at_topk = torch.exp(student_log_probs_at_topk)
            
            # Compute per-token KL: sum over k of p_teacher * (log p_teacher - log p_student)
            per_token_kl = (teacher_probs_chunk * (teacher_log_probs_chunk - student_log_probs_at_topk)).sum(dim=-1)
            
            # Compute per-token entropy: -sum over k of p_student * log p_student
            per_token_entropy = -(student_probs_at_topk * student_log_probs_at_topk).sum(dim=-1)
            
            # Apply mask if present
            if chunk_mask is not None:
                per_token_kl = per_token_kl * chunk_mask.float()
                per_token_entropy = per_token_entropy * chunk_mask.float()
            
            total_kl = total_kl + per_token_kl.sum()
            total_entropy = total_entropy + per_token_entropy.sum()
        
        # Return sum and count for proper CP aggregation in loss_func
        return total_kl, total_entropy, num_valid

    def loss_func(
        self,
        output_tensor: torch.Tensor,
        *,
        labels: torch.Tensor,
        assistant_mask: torch.Tensor,
        teacher_top_k_tokens: torch.Tensor,
        teacher_top_k_log_probs: torch.Tensor,
        teacher_mask: Optional[torch.Tensor] = None,
    ):
        """Compute sparse distillation loss.
        
        Args:
            output_tensor: Student logits [batch, seq_len, vocab_size]
            labels: Token labels [batch, seq_len] (-100 for non-loss positions)
            assistant_mask: Boolean mask for assistant tokens [batch, seq_len]
            teacher_top_k_tokens: Top-k token indices [batch, num_assistant, k]
            teacher_top_k_log_probs: Top-k log probs [batch, num_assistant, k]
            teacher_mask: Valid teacher position mask [batch, num_assistant]
        """
        args = get_args()
        student_logits = output_tensor
        
        # For next-token prediction, shift the mask: predict token at position i using logits at i-1
        # assistant_mask[1:] tells us which positions we predict (where loss is computed)
        shifted_mask = assistant_mask[:, 1:]  # [batch, seq_len - 1]
        student_logits_for_loss = student_logits[:, :-1]  # [batch, seq_len - 1, vocab]
        
        # Extract student logits only at assistant positions
        # Flatten for easier indexing
        batch_size, seq_len_minus_1, vocab_size = student_logits_for_loss.shape
        student_logits_flat = student_logits_for_loss.reshape(-1, vocab_size)  # [batch * (seq-1), vocab]
        shifted_mask_flat = shifted_mask.reshape(-1)  # [batch * (seq-1)]
        
        # Get indices of assistant positions
        assistant_indices = shifted_mask_flat.nonzero(as_tuple=True)[0]
        student_logits_sparse = student_logits_flat[assistant_indices]  # [num_assistant_total, vocab]
        
        # Flatten teacher data (already sparse, one entry per assistant token)
        # teacher_top_k_tokens: [batch, max_num_assistant, k] -> [num_assistant_total, k]
        # We need to handle padding in teacher data
        if teacher_mask is not None:
            teacher_mask_flat = teacher_mask.reshape(-1)
            teacher_valid_indices = teacher_mask_flat.nonzero(as_tuple=True)[0]
            teacher_top_k_tokens_flat = teacher_top_k_tokens.reshape(-1, teacher_top_k_tokens.shape[-1])
            teacher_top_k_log_probs_flat = teacher_top_k_log_probs.reshape(-1, teacher_top_k_log_probs.shape[-1])
            teacher_top_k_tokens_sparse = teacher_top_k_tokens_flat[teacher_valid_indices]
            teacher_top_k_log_probs_sparse = teacher_top_k_log_probs_flat[teacher_valid_indices]
            sparse_teacher_mask = None  # Already filtered out invalid
        else:
            teacher_top_k_tokens_sparse = teacher_top_k_tokens.reshape(-1, teacher_top_k_tokens.shape[-1])
            teacher_top_k_log_probs_sparse = teacher_top_k_log_probs.reshape(-1, teacher_top_k_log_probs.shape[-1])
            sparse_teacher_mask = None
        
        # Compute distillation loss (returns sum and count for CP aggregation)
        kl_loss_sum, entropy_sum, kl_loss_count = self.sparse_topk_kl_div(
            student_logits_sparse,
            teacher_top_k_tokens_sparse,
            teacher_top_k_log_probs_sparse,
            sparse_teacher_mask,
        )
        
        # All-reduce KL loss across CP group if needed
        if args.context_parallel_size > 1:
            kl_stats = torch.stack([kl_loss_sum, entropy_sum, kl_loss_count])
            torch.distributed.all_reduce(
                kl_stats, op=torch.distributed.ReduceOp.SUM, 
                group=mpu.get_context_parallel_group()
            )
            kl_loss_sum, entropy_sum, kl_loss_count = kl_stats[0], kl_stats[1], kl_stats[2]
        
        kl_loss = kl_loss_sum / kl_loss_count.clamp(min=1)
        entropy = entropy_sum / kl_loss_count.clamp(min=1)
        
        loss = self.distill_alpha * kl_loss
        
        metric = {
            'kl_loss': kl_loss.detach().clone(),
            'entropy': entropy.detach().clone(),
        }

        with torch.set_grad_enabled(self.sft_alpha > 0): 
            # Use standard cross-entropy loss on all tokens where labels != -100
            # logits[:, :-1] predicts tokens at positions 1 to seq-1
            # labels[:, :-1] contains targets for positions 0 to seq-2, which are tokens 1 to seq-1
            # (In Megatron convention, labels[i] = target token to predict from position i)
            logits_for_sft = student_logits_for_loss.transpose(0, 1).contiguous()  # [seq-1, batch, vocab]
            labels_for_sft = labels[:, :-1]  # [batch, seq-1] - aligned with logits
            
            per_token_loss = self.unwrapped_models[0].compute_language_model_loss(
                labels_for_sft, logits_for_sft
            )
            
            loss_mask = labels_for_sft != -100
            sft_loss_sum = (per_token_loss * loss_mask).sum()
            sft_loss_count = loss_mask.sum().float()
            
            # All-reduce across CP group if needed
            if args.context_parallel_size > 1:
                sft_stats = torch.stack([sft_loss_sum, sft_loss_count])
                torch.distributed.all_reduce(
                    sft_stats, op=torch.distributed.ReduceOp.SUM, 
                    group=mpu.get_context_parallel_group()
                )
                sft_loss_sum, sft_loss_count = sft_stats[0], sft_stats[1]
            
            sft_loss = sft_loss_sum / sft_loss_count.clamp(min=1)
            metric['sft_loss'] = sft_loss.detach().clone()
            
        # Add SFT loss if enabled
        if self.sft_alpha > 0:
            loss = loss + self.sft_alpha * sft_loss
        
        metric['loss'] = loss.detach().clone()
        metric = self._all_reduce_metric(metric)
        
        # Normalize by CP size for gradient scaling
        loss = loss / mpu.get_context_parallel_world_size()
        
        return loss, metric

    def forward_step(self, data_iterator, model):
        """Forward step for sparse distillation."""
        timers = get_timers()
        
        unwrapped_model = model.module.module
        vp_stage = unwrapped_model.vp_stage
        
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        
        # Extract distillation-specific data
        labels = data.pop('labels', None)
        assistant_mask = data.pop('assistant_mask', None)
        teacher_top_k_tokens = data.pop('teacher_top_k_tokens', None)
        teacher_top_k_log_probs = data.pop('teacher_top_k_log_probs', None)
        teacher_mask = data.pop('teacher_mask', None)
        data.pop('loss_scale', None)
        
        with self.stimer:
            output = model(**data)
        
        return output, partial(
            self.loss_func,
            labels=labels,
            assistant_mask=assistant_mask,
            teacher_top_k_tokens=teacher_top_k_tokens,
            teacher_top_k_log_probs=teacher_top_k_log_probs,
            teacher_mask=teacher_mask,
        )
