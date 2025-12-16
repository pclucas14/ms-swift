# Copyright (c) Alibaba, Inc. and its affiliates.
"""Configuration for Role-based MoE training."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RoleBasedMoEConfig:
    """Configuration for Role-based MoE training.

    This configuration controls the behavior of role-based routing in MoE models,
    where tokens are routed to experts based on their conversation role (system,
    user, assistant) rather than learned gating functions.

    Attributes:
        enabled: Whether to enable role-based routing. If False, standard TopK routing is used.
        role_to_expert: Mapping from role names to expert indices. Default maps system/user to
                       expert 0 and assistant to expert 1.
        router_aux_loss_coef: Coefficient for router auxiliary loss (load balancing).
        strict_role_checking: If True, raises an error when unexpected roles are encountered.
    """

    enabled: bool = True

    # Role to expert mapping
    role_to_expert: Dict[str, int] = field(default_factory=lambda: {
        'system': 0,
        'user': 0,
        'assistant': 1,
    })

    # Router configuration
    router_aux_loss_coef: float = 0.001

    # Validation
    strict_role_checking: bool = True

    def get_expert_for_role(self, role: str) -> int:
        """Get the expert index for a given role.

        Args:
            role: Role name ('system', 'user', or 'assistant')

        Returns:
            Expert index for the given role

        Raises:
            ValueError: If role is not found and strict_role_checking is True
        """
        if role in self.role_to_expert:
            return self.role_to_expert[role]

        if self.strict_role_checking:
            raise ValueError(
                f"Unknown role '{role}'. Expected one of: {list(self.role_to_expert.keys())}"
            )

        # Default to expert 0 for unknown roles
        return 0

    def validate(self, num_experts: int):
        """Validate configuration against model architecture.

        Args:
            num_experts: Number of experts in the model

        Raises:
            ValueError: If configuration is invalid for the given model
        """
        max_expert_idx = max(self.role_to_expert.values())
        if max_expert_idx >= num_experts:
            raise ValueError(
                f"Role mapping requires expert index {max_expert_idx}, "
                f"but model only has {num_experts} experts"
            )
