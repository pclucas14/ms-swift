# Copyright (c) Alibaba, Inc. and its affiliates.
"""Trainer implementation for Role-based MoE models."""

from functools import partial
from typing import Optional

import torch
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.training import get_args, get_timers, print_rank_0

from swift.megatron.trainers import MegatronTrainer
from swift.utils import deep_getattr, get_logger

from .config import RoleBasedMoEConfig
from .router import RoleBasedTopKRouter

logger = get_logger()


class RoleBasedMoETrainer(MegatronTrainer):
    """Trainer for Role-based MoE models.

    This trainer extends MegatronTrainer to support role-based routing in MoE models.
    Instead of learned routing, tokens are assigned to experts based on their
    conversation role (system, user, assistant).

    Key features:
    - Automatic router replacement: Replaces standard TopKRouter with RoleBasedTopKRouter
    - Role injection: Automatically passes role information to routers during training
    - Validation: Ensures role information is present when required

    Usage:
        config = RoleBasedMoEConfig(enabled=True)
        trainer = RoleBasedMoETrainer(args, template, role_moe_config=config)
        trainer.train(train_dataset, val_dataset, data_collator)
    """

    def __init__(
        self,
        args,
        template,
        role_moe_config: Optional[RoleBasedMoEConfig] = None
    ):
        """Initialize the Role-based MoE trainer.

        Args:
            args: Training arguments
            template: Template instance (should support role tracking if role_moe_config.enabled)
            role_moe_config: Configuration for role-based MoE. If None, creates default config.
        """
        super().__init__(args, template)

        # Initialize role-based MoE configuration
        self.role_moe_config = role_moe_config or RoleBasedMoEConfig()

        # Track whether routers have been replaced
        self._routers_replaced = False

        if self.role_moe_config.enabled:
            logger.info("Role-based MoE enabled")
            logger.info(f"Role to expert mapping: {self.role_moe_config.role_to_expert}")
        else:
            logger.info("Role-based MoE disabled, using standard TopK routing")

    def _setup_role_based_router(self, model):
        """Replace TopKRouter with RoleBasedTopKRouter in the model.

        This method traverses the model and replaces all instances of TopKRouter
        with RoleBasedTopKRouter to enable role-based routing.

        Args:
            model: The model to modify
        """
        if self._routers_replaced:
            logger.debug("Routers already replaced, skipping")
            return

        pg = get_default_model_comm_pgs()
        replaced_count = 0

        for name, module in model.named_modules():
            if isinstance(module, TopKRouter):
                old_router = module

                # Create new role-based router with same config
                new_router = RoleBasedTopKRouter(
                    config=old_router.config,
                    model_comm_pgs=pg
                )

                # Replace in parent module
                module_idx = name.split('.')[-1]
                parent_module_name = '.'.join(name.split('.')[:-1])

                if parent_module_name:
                    parent_module = deep_getattr(model, parent_module_name)
                else:
                    parent_module = model

                setattr(parent_module, module_idx, new_router)

                replaced_count += 1
                print_rank_0(f'Replaced TopKRouter with RoleBasedTopKRouter in {name}')

                # Validate configuration
                try:
                    self.role_moe_config.validate(old_router.config.num_moe_experts)
                except ValueError as e:
                    logger.error(f"Role-based MoE configuration invalid: {e}")
                    raise

        if replaced_count == 0:
            logger.warning(
                "No TopKRouter modules found in model. "
                "Are you sure this is an MoE model?"
            )
        else:
            logger.info(f"Successfully replaced {replaced_count} routers")

        self._routers_replaced = True

    def setup_model_and_optimizer(self, model_provider_func, model_type, *args, **kwargs):
        """Override to inject role-based router setup.

        This method wraps the model provider function to automatically replace
        routers after model creation.

        Args:
            model_provider_func: Function that creates the model
            model_type: Type of model (encoder/decoder)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (model, optimizer, opt_param_scheduler)
        """

        def role_aware_model_provider(*_args, **_kwargs):
            """Wrapper that adds role-based routing after model creation."""
            model = model_provider_func(*_args, **_kwargs)

            # Setup role-based routing if enabled
            if self.role_moe_config.enabled:
                self._setup_role_based_router(model)

            return model

        # Call parent with wrapped provider
        return super().setup_model_and_optimizer(
            role_aware_model_provider, model_type, *args, **kwargs
        )

    def _set_router_roles(self, roles: torch.Tensor):
        """Set role information on all RoleBasedTopKRouter modules.

        Args:
            roles: Tensor of shape [batch_size, seq_len] containing role IDs
        """
        if roles is None:
            logger.warning("Attempted to set None roles on routers")
            return

        for model in self.wrapped_models:
            for name, module in model.named_modules():
                if isinstance(module, RoleBasedTopKRouter):
                    module.set_token_roles(roles)
                    logger.debug(f"Set role_ids on {name}")

    def get_batch(self, data_iterator, vp_stage=None):
        """Override to extract and inject role information.

        This method extracts role information from the batch and passes it
        to the routers before the forward pass.

        IMPORTANT: The 'roles' key is ALWAYS removed from the batch, regardless
        of whether role-based MoE is enabled. This prevents the 'roles' key from
        being passed to the model's forward method where it is not expected.

        Args:
            data_iterator: Iterator over batches
            vp_stage: Virtual pipeline stage (for pipeline parallelism)

        Returns:
            Batch dictionary with role information removed (handled separately)
        """
        batch = super().get_batch(data_iterator, vp_stage)

        # ALWAYS pop 'roles' from batch to prevent it from being passed to model
        # The model's forward method doesn't expect this key
        roles = batch.pop('roles', None)

        # Only use roles if role-based MoE is enabled
        if self.role_moe_config.enabled:
            if roles is not None:
                self._set_router_roles(roles)
            else:
                # This is a critical error - role-based MoE needs role information
                if self.role_moe_config.strict_role_checking:
                    raise RuntimeError(
                        "Role information missing from batch but role-based MoE is enabled. "
                        "Ensure your template supports role tracking (use RoleTrackingMixin)."
                    )
                else:
                    logger.warning(
                        "Role information missing from batch. "
                        "Role-based routing may not work correctly."
                    )
        # If role-based MoE is disabled and roles are present, just discard them
        elif roles is not None:
            logger.debug("Discarding role information (role-based MoE disabled)")

        return batch

    def forward_step(self, data_iterator, model):
        """Override to ensure role information is set before forward pass.

        Args:
            data_iterator: Iterator over batches
            model: Model to run forward pass on

        Returns:
            Tuple of (output_tensor, loss_func)
        """
        timers = get_timers()

        # Get the batch
        vp_stage = model.module.module.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)

        # Note: Role information is already set in get_batch()
        # The roles have been popped from data and set on the routers

        timers('batch-generator').stop()
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')

        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)

        with self.stimer:
            output_tensor = model(**data)

        packed_seq_params = data.get('packed_seq_params')

        if self.args.task_type == 'seq_cls':
            loss_func = partial(
                self.seq_cls_loss_func,
                labels=labels,
                packed_seq_params=packed_seq_params
            )
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params
            )

        return output_tensor, loss_func
