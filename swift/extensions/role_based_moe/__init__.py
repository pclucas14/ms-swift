# Copyright (c) Alibaba, Inc. and its affiliates.
"""Role-based MoE extension for ms-swift.

This extension provides role-based routing for Mixture of Experts (MoE) models,
where tokens are routed to experts based on their conversation role (system,
user, assistant) rather than learned gating mechanisms.

Two approaches are available:

1. **Minimal Token-Based Approach (RECOMMENDED)**:
   - Extracts roles from token IDs using special role markers
   - No template modification required
   - Simpler and less invasive
   - Use RoleBasedMoETrainer

2. **Mixin-Based Approach**:
   - Tracks roles during template encoding
   - Requires template patching
   - More complex but works if tokens don't have clear role markers
   - Use RoleBasedMoETrainer + create_role_tracking_template

Key components:
- RoleBasedMoEConfig: Configuration for role-based routing
- RoleBasedTopKRouter: Router that uses roles instead of learned gates
- RoleBasedMoETrainer: Token-based trainer (recommended)
- RoleBasedMoETrainer: Mixin-based trainer (alternative)
- TokenBasedRoleExtractor: Extracts roles from token IDs

Example usage (Minimal Approach):
    >>> from swift.extensions.role_based_moe import (
    ...     RoleBasedMoEConfig,
    ...     RoleBasedMoETrainer,
    ... )
    >>>
    >>> # Configure role-based MoE
    >>> config = RoleBasedMoEConfig(
    ...     enabled=True,
    ...     role_to_expert={'system': 0, 'user': 0, 'assistant': 1}
    ... )
    >>>
    >>> # Create and use trainer (no template modification needed!)
    >>> trainer = RoleBasedMoETrainer(args, template, config, template_type='qwen')
    >>> trainer.train(train_dataset, val_dataset, data_collator)

Example usage (Mixin Approach):
    >>> from swift.extensions.role_based_moe import (
    ...     RoleBasedMoEConfig,
    ...     RoleBasedMoETrainer,
    ...     create_role_tracking_template
    ... )
    >>> from swift.llm.template import Template
    >>>
    >>> # Create a role-tracking template
    >>> RoleTrackingTemplate = create_role_tracking_template(Template)
    >>> template = RoleTrackingTemplate(...)
    >>>
    >>> # Configure role-based MoE
    >>> config = RoleBasedMoEConfig(
    ...     enabled=True,
    ...     role_to_expert={'system': 0, 'user': 0, 'assistant': 1}
    ... )
    >>>
    >>> # Create and use trainer
    >>> trainer = RoleBasedMoETrainer(args, template, config)
    >>> trainer.train(train_dataset, val_dataset, data_collator)
"""

from .config import RoleBasedMoEConfig
from .router import RoleBasedTopKRouter, RoleEnum
from .token_based_role_extraction import TokenBasedRoleExtractor, create_role_extractor
from .trainer import RoleBasedMoETrainer

__all__ = [
    'RoleBasedMoEConfig',
    'RoleBasedTopKRouter',
    'RoleEnum',
    'RoleTrackingMixin',
    'create_role_tracking_template',
    'TokenBasedRoleExtractor',
    'create_role_extractor',
    'RoleBasedMoETrainer',
]

__version__ = '0.1.0'
