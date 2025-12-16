# Copyright (c) Alibaba, Inc. and its affiliates.
"""Role-based MoE extension for ms-swift.

This extension provides role-based routing for Mixture of Experts (MoE) models,
where tokens are routed to experts based on their conversation role (system,
user, assistant) rather than learned gating mechanisms.

Key components:
- RoleBasedMoEConfig: Configuration for role-based routing
- RoleBasedTopKRouter: Router that uses roles instead of learned gates
- RoleTrackingMixin: Mixin to add role tracking to templates
- RoleBasedMoETrainer: Trainer that handles role-based MoE models

Example usage:
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
from .template import RoleTrackingMixin, create_role_tracking_template
from .trainer import RoleBasedMoETrainer

__all__ = [
    'RoleBasedMoEConfig',
    'RoleBasedTopKRouter',
    'RoleEnum',
    'RoleTrackingMixin',
    'create_role_tracking_template',
    'RoleBasedMoETrainer',
]

__version__ = '0.1.0'
