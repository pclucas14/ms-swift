# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pipeline for sparse top-k distillation training."""
from functools import partial
from typing import List, Optional, Union

from swift.megatron.arguments import MegatronSparseDistillArguments
from swift.megatron.datasets import SparseDistillDataset, sparse_distill_data_collator
from swift.megatron.trainers import MegatronSparseDistillTrainer
from swift.megatron.utils import get_padding_to
from swift.utils import get_logger, is_last_rank
from .sft import MegatronSft

logger = get_logger()


class MegatronSparseDistill(MegatronSft):
    """Pipeline for sparse top-k distillation training.
    
    Uses pre-computed teacher top-k probabilities stored in .pt shard files.
    """
    args_class = MegatronSparseDistillArguments
    args: args_class

    def prepare_trainer(self):
        return MegatronSparseDistillTrainer(self.args, self.template)

    def _prepare_dataset(self):
        """Load sparse distillation dataset from .pt shard files."""
        args = self.args
        
        logger.info(f'Loading sparse distillation data from: {args.sparse_distill_data}')
        
        train_dataset = SparseDistillDataset(
            data_path=args.sparse_distill_data,
            max_length=args.max_length,
        )
        
        # No validation dataset for now (could split if needed)
        val_dataset = None
        
        logger.info(f'Loaded {len(train_dataset)} training samples')
        
        return train_dataset, val_dataset

    def _get_data_collator(self):
        """Get data collator for sparse distillation."""
        padding_to = get_padding_to(self.args)
        
        # CP uses 2*cp_size chunks for load balancing, so ensure padding is multiple of 2*cp_size
        cp_size = getattr(self.args, 'context_parallel_size', 1)
        if cp_size > 1:
            cp_padding = 2 * cp_size
            if padding_to is None:
                padding_to = cp_padding
            else:
                # Ensure padding_to is a multiple of 2*cp_size
                padding_to = ((padding_to + cp_padding - 1) // cp_padding) * cp_padding
        
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        logger.info(f'Using sparse_distill_data_collator with padding_to={padding_to} (CP={cp_size})')
        
        return partial(
            sparse_distill_data_collator,
            padding_to=padding_to,
            pad_token_id=pad_token_id,
        )


def megatron_sparse_distill_main(args: Optional[Union[List[str], MegatronSparseDistillArguments]] = None):
    """Main entry point for sparse distillation training."""
    return MegatronSparseDistill(args).main()
