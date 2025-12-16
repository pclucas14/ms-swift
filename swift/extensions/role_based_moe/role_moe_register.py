# Copyright (c) Alibaba, Inc. and its affiliates.
"""Custom registration for Role-based MoE models.

This module patches MegatronSft.prepare_trainer() to return RoleBasedMoETrainer
instead of the standard MegatronTrainer. This is a clean approach that doesn't
require monkey-patching the trainer class itself.

Usage:
    Add to your training script:
    --custom_register_path /path/to/role_moe_register.py
"""

print("="*80)
print("CUSTOM REGISTER LOADING - START")
print("="*80)

from typing import Any, Dict

from swift.llm import get_model_tokenizer_with_flash_attn
from swift.llm.model.utils import ModelInfo
from swift.utils import get_logger

# Import the role-based MoE extension
from swift.extensions.role_based_moe import (
    RoleBasedMoEConfig,
    RoleBasedMoETrainer as RoleBasedMoETrainer,
)

logger = get_logger()


def register_role_based_trainer():
    """Patch MegatronSft to use RoleBasedMoETrainer.

    This patches the prepare_trainer() method of MegatronSft to return
    RoleBasedMoETrainer instead of MegatronTrainer.
    """
    try:
        from swift.megatron.train.sft import MegatronSft

        # Store original method
        _original_prepare_trainer = MegatronSft.prepare_trainer

        def prepare_role_based_trainer(self):
            """Override prepare_trainer to use RoleBasedMoETrainer."""

            # Create role-based MoE configuration
            role_moe_config = RoleBasedMoEConfig(
                enabled=True,
                role_to_expert={
                    'system': 0,
                    'user': 0,
                    'assistant': 1,
                },
                router_aux_loss_coef=0.001,
                strict_role_checking=True
            )

            logger.info("="*60)
            logger.info("Using RoleBasedMoETrainer (via patched prepare_trainer)")
            logger.info(f"Role mapping: {role_moe_config.role_to_expert}")
            logger.info("="*60)

            # Return our custom trainer
            return RoleBasedMoETrainer(self.args, self.template, role_moe_config)

        # Patch the method
        MegatronSft.prepare_trainer = prepare_role_based_trainer

        logger.info("✓ Successfully patched MegatronSft.prepare_trainer()")
        logger.info("  Will use RoleBasedMoETrainer instead of MegatronTrainer")

    except Exception as e:
        logger.error(f"Failed to patch MegatronSft: {e}")
        logger.warning("Falling back to standard trainer")
        import traceback
        traceback.print_exc()

def get_role_based_moe_model(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    """Custom model loading function for role-based MoE.

    Args:
        model_dir: Path to model directory
        model_info: Model information
        model_kwargs: Model keyword arguments
        load_model: Whether to load model weights
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading role-based MoE model from {model_dir}")

    # Load model using standard function
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir=model_dir,
        model_info=model_info,
        model_kwargs=model_kwargs,
        load_model=load_model,
        **kwargs
    )

    if model is not None:
        logger.info(f"Model loaded: {model.__class__.__name__}")

        # Count MoE layers
        moe_layer_count = 0
        for name, module in model.named_modules():
            # Check for router or experts indicating MoE layer
            if hasattr(module, 'gate') or (hasattr(module, 'router') and hasattr(module, 'experts')):
                moe_layer_count += 1

        if moe_layer_count > 0:
            logger.info(f"Found {moe_layer_count} MoE layers in model")
            logger.info("Routers will be replaced with RoleBasedTopKRouter during training")
        else:
            logger.warning("No MoE layers found in model. Are you sure this is an MoE model?")

    return model, tokenizer


# Perform registration when module is imported
logger.info("="*60)
logger.info("Initializing Role-based MoE Custom Registration")
logger.info("="*60)

# Register custom trainer (by patching MegatronSft.prepare_trainer)
register_role_based_trainer()

logger.info("="*60)
logger.info("Role-based MoE registration complete")
logger.info("="*60)
logger.info("")
logger.info("Configuration:")
logger.info("  • Trainer: RoleBasedMoETrainer (via MegatronSft.prepare_trainer)")
logger.info("  • Template: Role-tracking enabled")
logger.info("  • Role mapping: system/user → expert 0, assistant → expert 1")
logger.info("  • Strict checking: enabled")
logger.info("")

print("="*80)
print("CUSTOM REGISTER LOADING - COMPLETE")
print("="*80)
