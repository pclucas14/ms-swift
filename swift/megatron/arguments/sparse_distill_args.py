# Copyright (c) ModelScope Contributors. All rights reserved.
"""Arguments for sparse top-k distillation training."""
from dataclasses import dataclass, field
from typing import Optional

from swift.utils import get_logger, to_abspath
from .sft_args import MegatronSftArguments

logger = get_logger()


@dataclass
class MegatronSparseDistillArguments(MegatronSftArguments):
    """Arguments for sparse top-k distillation training.
    
    Uses pre-computed teacher top-k probabilities from .pt shard files.
    """
    
    # Path to sparse distillation data (directory with .pt shards or single .pt file)
    sparse_distill_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to sparse distillation data (directory with .pt shards)'}
    )
    
    # Loss weights
    distill_alpha: float = field(
        default=1.0,
        metadata={'help': 'Weight for KL divergence distillation loss'}
    )
    sft_alpha: float = field(
        default=0.0,
        metadata={'help': 'Weight for SFT cross-entropy loss (0 = pure distillation)'}
    )
    
    # Temperature for distillation
    temperature: float = field(
        default=1.0,
        metadata={'help': 'Temperature for softmax in distillation loss'}
    )

    def __post_init__(self):
        # Convert path to absolute
        if self.sparse_distill_data is not None:
            self.sparse_distill_data = to_abspath(self.sparse_distill_data, check_path_exist=True)
        
        # Skip dataset validation for sparse distill (we don't use standard datasets)
        # Temporarily set a dummy dataset to pass parent validation
        original_dataset = self.dataset
        original_cached = self.cached_dataset
        if not self.dataset and not self.cached_dataset:
            self.dataset = ['__sparse_distill_dummy__']
        
        # Initialize parent (skip model initialization)
        self.load = to_abspath(self.load, check_path_exist=True)
        
        # Call grandparent's __post_init__ directly to avoid dataset check
        from .megatron_base_args import MegatronBaseArguments
        MegatronBaseArguments.__post_init__(self)
        
        # Restore original dataset values
        self.dataset = original_dataset
        self.cached_dataset = original_cached
        
        # Initialize save directory
        self._init_save()
        if self.tensorboard_dir is None and self.save is not None:
            self.tensorboard_dir = f'{self.save}/runs'
        self.tensorboard_dir = to_abspath(self.tensorboard_dir)
        
        # Validate sparse distill data path
        if self.sparse_distill_data is None:
            raise ValueError('--sparse_distill_data is required for sparse distillation training')
        
        logger.info(f'Sparse distillation config: data={self.sparse_distill_data}, '
                   f'distill_alpha={self.distill_alpha}, sft_alpha={self.sft_alpha}, '
                   f'temperature={self.temperature}')
