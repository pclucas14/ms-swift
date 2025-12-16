# Copyright (c) Alibaba, Inc. and its affiliates.
"""Role-based router for MoE models."""

from typing import Optional, Tuple

import torch
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import ModelCommProcessGroups

from swift.utils import get_logger

logger = get_logger()


class RoleEnum:
    """Enumeration of conversation roles."""
    system: int = 0
    user: int = 1
    assistant: int = 2


class RoleBasedTopKRouter(TopKRouter):
    """Router that uses conversation roles instead of learned routing.

    This router assigns tokens to experts based on their conversation role
    (system, user, assistant) rather than using a learned gating mechanism.
    This is useful for creating specialized experts that handle different
    parts of the conversation.

    The default mapping is:
    - system/user tokens -> expert 0
    - assistant tokens -> expert 1
    - padding tokens -> expert 0

    Attributes:
        token_routing: Tensor of shape [seq_len, batch_size] containing role IDs
                      for each token in the current batch.
    """

    def __init__(
        self,
        config: TransformerConfig,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None
    ) -> None:
        """Initialize the role-based router.

        Args:
            config: Transformer configuration containing MoE settings
            model_comm_pgs: Model communication process groups for distributed training
        """
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        self.token_routing = None
        logger.info(f"Initialized RoleBasedTopKRouter with {self.num_experts} experts")

    def gating(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pass-through gating function (no learned gate).

        In role-based routing, we don't use a learned gating mechanism.
        Instead, we return the inputs directly and handle routing based
        on pre-assigned roles.

        Args:
            inputs: Hidden states tensor

        Returns:
            The input tensor unchanged
        """
        return inputs

    def set_token_roles(self, token_roles: torch.Tensor):
        """Set role IDs for the current batch.

        Args:
            token_roles: Tensor of shape [batch_size, seq_len] containing role IDs
                        where:
                        - 0 = system role
                        - 1 = user role
                        - 2 = assistant role
                        - -1 = padding token
        """
        if token_roles is None:
            logger.warning("Received None for token_roles, skipping role assignment")
            return

        # Map roles to expert indices
        # system (0) and user (1) -> expert 0
        # assistant (2) -> expert 1
        system_and_user = (token_roles == RoleEnum.system) | (token_roles == RoleEnum.user)
        assistant = (token_roles == RoleEnum.assistant)

        # Validate that all non-padding tokens have valid roles
        padding = (token_roles == -1)
        valid_roles = system_and_user | assistant | padding

        if not valid_roles.all():
            invalid_count = (~valid_roles).sum().item()
            logger.warning(
                f"Found {invalid_count} tokens with unexpected role IDs. "
                "These will be assigned to expert 0."
            )

        # Create expert assignment tensor
        # setting to -1 to ensure that all tokens are replaced.
        roles = token_roles.new_zeros(token_roles.size()) - 1
        roles[system_and_user] = 0
        roles[assistant] = 1
        roles[~valid_roles] = 0  # Assign unexpected roles to expert 0
        
        self.token_routing = roles

        logger.debug(
            f"Set token roles: {system_and_user.sum()} system/user tokens, "
            f"{assistant.sum()} assistant tokens, {padding.sum()} padding tokens"
        )

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform role-based routing.

        Args:
            logits: Tensor of shape [seq_len, batch_size, hidden_dim]
                   Note: In role-based routing, these are actually the hidden states,
                   not logits, since we bypass the learned gating mechanism.

        Returns:
            probs: Tensor of shape [num_tokens, topk] with routing probabilities.
                  All weight goes to the selected expert (probability = 1.0).
            routing_map: Boolean tensor of shape [num_tokens, num_experts] indicating
                        which expert(s) each token is assigned to.

        Raises:
            RuntimeError: If token_routing has not been set via set_token_roles()
        """

        if self.token_routing is None:
            raise RuntimeError(
                "Token roles have not been set. Call set_token_roles() before routing."
            )

        seq_len, bs, _ = logits.size()
        num_tokens = bs * seq_len

        # Initialize routing map (all False)
        routing_map = torch.zeros(
            (num_tokens, self.num_experts),
            dtype=torch.bool,
            device=logits.device
        )

        # Flatten token roles [seq_len, bs] -> [num_tokens]
        token_roles_flat = self.token_routing.view(-1)

        # Assign tokens to experts based on roles
        for expert_idx in range(min(2, self.num_experts)):
            mask = (token_roles_flat == expert_idx)
            routing_map[mask, expert_idx] = True

        # Handle padding tokens (-1) -> assign to expert 0
        padding_mask = (token_roles_flat == -1)
        routing_map[padding_mask, 0] = True

        # Create uniform probabilities for selected experts
        # All weight goes to the single selected expert
        probs = torch.zeros(
            (num_tokens, self.topk),
            dtype=logits.dtype,
            device=logits.device
        )
        probs[:, 0] = 1.0  # 100% weight to selected expert

        # Log routing statistics
        if logger.isEnabledFor(10):  # DEBUG level
            for expert_idx in range(self.num_experts):
                count = routing_map[:, expert_idx].sum().item()
                logger.debug(f"Expert {expert_idx}: {count} tokens assigned")

        return probs, routing_map
