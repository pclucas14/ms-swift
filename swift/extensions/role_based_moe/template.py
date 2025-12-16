# Copyright (c) Alibaba, Inc. and its affiliates.
"""Template extensions for role tracking in Role-based MoE."""

from typing import List, Optional, Tuple, Union

from swift.utils import get_logger

logger = get_logger()


class RoleEnum:
    """Enumeration of conversation roles for role-based routing."""
    system: int = 0
    user: int = 1
    assistant: int = 2


class RoleTrackingMixin:
    """Mixin to add role tracking capabilities to any Template.

    This mixin extends the Template class to track conversation roles
    (system, user, assistant) for each token during encoding. This is
    required for role-based MoE routing where tokens are assigned to
    experts based on their role in the conversation.

    The mixin overrides the _encode_context_list method to add role
    tracking while maintaining compatibility with the base Template API.

    Usage:
        class MyRoleTrackingTemplate(RoleTrackingMixin, Template):
            pass
    """

    def _encode_context_list_with_roles(
        self,
        context_list: List[Union[str, List[int]]],
        loss_scale_list: Optional[List[float]] = None
    ) -> Tuple[List[int], List[int], Optional[List[float]], List[int]]:
        """Encode context list and track roles for each token.

        This method extends the base _encode_context_list by adding role tracking.
        It assigns a role ID to each token based on its position in the conversation:
        - First context (system prompt): RoleEnum.system
        - Odd contexts (user messages): RoleEnum.user
        - Even contexts (assistant responses): RoleEnum.assistant

        Args:
            context_list: List of context strings or token lists to encode
            loss_scale_list: Optional list of loss scale values for each context

        Returns:
            Tuple of (input_ids, labels, loss_scale, roles) where:
            - input_ids: List of token IDs
            - labels: List of label IDs (same as input_ids but with -100 for non-training tokens)
            - loss_scale: Optional list of loss scale values per token
            - roles: List of role IDs per token
        """
        # First, call the base implementation to get input_ids, labels, and loss_scale
        # We need to temporarily disable role tracking in the base method
        result = self._encode_context_list_base(context_list, loss_scale_list)

        if len(result) == 3:
            input_ids, labels, loss_scale = result
        else:
            # Handle case where base method already returns roles (shouldn't happen)
            input_ids, labels, loss_scale, _ = result

        # Now add role tracking
        roles: List[int] = []

        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)

        for i, context in enumerate(context_list):
            # Determine role based on position in conversation
            if i == 0:
                expected_role = RoleEnum.system
            elif i % 2 == 0:
                expected_role = RoleEnum.user
            else:
                expected_role = RoleEnum.assistant

            # Get token count for this context
            if isinstance(context, str):
                token_list = self._tokenize(context)
            else:
                token_list = context

            # Assign role to all tokens in this context
            roles.extend([expected_role] * len(token_list))

        # Validate that roles match input_ids length
        if len(roles) != len(input_ids):
            logger.warning(
                f"Role tracking mismatch: {len(roles)} roles for {len(input_ids)} tokens. "
                "Padding roles to match."
            )
            # Pad with system role (most conservative choice)
            while len(roles) < len(input_ids):
                roles.append(RoleEnum.system)
            # Truncate if too many
            roles = roles[:len(input_ids)]

        return input_ids, labels, loss_scale, roles

    def _encode_context_list_base(
        self,
        context_list: List[Union[str, List[int]]],
        loss_scale_list: Optional[List[float]] = None
    ) -> Tuple[List[int], List[int], Optional[List[float]]]:
        """Call the original _encode_context_list from the parent class.

        This method dynamically calls the parent's _encode_context_list to avoid
        infinite recursion when using the mixin.
        """
        # Get the parent class's _encode_context_list
        # We need to skip our own class and the mixin
        for base_class in self.__class__.__mro__[1:]:
            if base_class.__name__ != 'RoleTrackingMixin' and hasattr(base_class, '_encode_context_list'):
                method = base_class._encode_context_list
                # Call it with self bound
                return method(self, context_list, loss_scale_list)

        # Fallback if we can't find the parent method
        raise RuntimeError(
            "Could not find parent _encode_context_list method. "
            "Ensure RoleTrackingMixin is used with a Template class."
        )


def create_role_tracking_template(template_cls):
    """Factory function to create a role-tracking version of any Template class.

    This function creates a new class that inherits from both RoleTrackingMixin
    and the provided template class, enabling role tracking for that template.

    Args:
        template_cls: A Template class to add role tracking to

    Returns:
        A new class with role tracking capabilities

    Example:
        >>> from swift.llm.template import Template
        >>> RoleTrackingTemplate = create_role_tracking_template(Template)
        >>> template = RoleTrackingTemplate(...)
    """
    class _RoleTrackingTemplate(RoleTrackingMixin, template_cls):
        """Dynamically created role-tracking template class."""

        def _encode_context_list(
            self,
            context_list: List[Union[str, List[int]]],
            loss_scale_list: Optional[List[float]] = None
        ):
            """Override to enable role tracking."""
            return self._encode_context_list_with_roles(context_list, loss_scale_list)

    _RoleTrackingTemplate.__name__ = f'RoleTracking{template_cls.__name__}'
    _RoleTrackingTemplate.__qualname__ = f'RoleTracking{template_cls.__qualname__}'

    return _RoleTrackingTemplate
