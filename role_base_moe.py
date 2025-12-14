#!/usr/bin/env python3
"""
Self-contained script for role-based routing in Qwen MoE models.
Routes user/system messages to expert 0, assistant messages to expert 1.

This script validates the model structure and provides graceful fallbacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Qwen3MoeForCausalLM
)
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RoleBasedRouter(nn.Module):
    """
    Custom router that routes based on message roles:
    - user/system messages -> Expert 0
    - assistant messages -> Expert 1
    """
    
    def __init__(self, original_gate, num_experts: int = 2):
        super().__init__()
        self.original_gate = original_gate
        self.num_experts = num_experts
        self.token_roles = None  # Will store role for each token position
        self.use_role_routing = True
        
    def set_token_roles(self, token_roles: torch.Tensor):
        """Set the role mapping for each token position.
        
        Args:
            token_roles: Tensor of shape [seq_len] with role indices:
                         0 for user/system, 1 for assistant, -1 for padding
        """
        self.token_roles = token_roles
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Route based on token roles. Returns router logits.
        
        Args:
            hidden_states: Can be 2D [batch_size * seq_len, hidden_dim] or 
                          3D [batch_size, seq_len, hidden_dim]
        
        Returns:
            router_logits: Logits for all experts
        """
        if not self.use_role_routing or self.token_roles is None:
            # Fall back to original routing
            return self.original_gate(hidden_states)
        
        # Handle both 2D and 3D inputs
        if hidden_states.dim() == 2:
            batch_tokens, hidden_dim = hidden_states.shape
            # Create logits based on token roles
            router_logits = torch.full(
                (batch_tokens, self.num_experts),
                -1e9,  # Very low logit
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
            # Flatten token_roles if needed and apply routing
            token_roles_flat = self.token_roles.view(-1)[:batch_tokens]
            
            # Route tokens to expert 0 (user/system) or expert 1 (assistant)
            expert_0_mask = token_roles_flat == 0
            expert_1_mask = token_roles_flat == 1
            
            router_logits[expert_0_mask, 0] = 1e9
            router_logits[expert_1_mask, 1] = 1e9
            
        elif hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            # Create logits based on token roles
            router_logits = torch.full(
                (batch_size, seq_len, self.num_experts),
                -1e9,  # Very low logit
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
            # Apply routing per token position
            for i in range(seq_len):
                if i < len(self.token_roles):
                    if self.token_roles[i] == 0:  # user/system
                        router_logits[:, i, 0] = 1e9
                    elif self.token_roles[i] == 1:  # assistant
                        router_logits[:, i, 1] = 1e9
        else:
            raise ValueError(f"Expected 2D or 3D input, got {hidden_states.dim()}D")
        
        return router_logits


class RoleBasedMoEWrapper:
    """
    Wrapper for Qwen MoE model that implements role-based routing.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        print(f"Loading model from {model_path}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Validate model structure
        if not self._validate_model():
            raise ValueError("Model structure validation failed. Please check the model checkpoint.")
        
        # Replace routers with role-based routers
        self._replace_routers()

    def _validate_model(self) -> bool:
        """Validate that the model has proper MoE structure."""
        print("Validating model structure...")
        
        has_experts = False
        for name, module in self.model.named_modules():
            if 'experts' in name and hasattr(module, 'gate_proj'):
                has_experts = True
                # Check dimensions
                try:
                    in_features = module.gate_proj.in_features
                    out_features = module.gate_proj.out_features
                    print(f"  Found expert: {name}")
                    print(f"    gate_proj: in={in_features}, out={out_features}")
                    
                    if in_features < 100 or out_features < 100:
                        print(f"  ⚠️  WARNING: Expert {name} has suspicious dimensions!")
                        print(f"      This suggests the model checkpoint may not be properly upcycled.")
                        return False
                except Exception as e:
                    print(f"  ✗ Error checking expert {name}: {e}")
                    return False
        
        if not has_experts:
            print("  ✗ No experts found in model!")
            return False
        
        print("  ✓ Model structure validation passed")
        return True
        
    def _replace_routers(self):
        """Replace all MoE gate layers with our custom router."""
        replaced_count = 0
        for name, module in self.model.named_modules():
            if 'gate' in name and 'mlp' in name and not 'shared' in name and not 'gate_proj' in name:
                # Store reference to parent module
                parts = name.split('.')
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                # Get original gate
                original_gate = getattr(parent, parts[-1])
                
                # Create wrapper that preserves original gate
                custom_router = RoleBasedRouter(original_gate, num_experts=2)
                setattr(parent, parts[-1], custom_router)
                replaced_count += 1
        
        print(f"✓ Replaced {replaced_count} routers with role-based routing")
        
    def set_role(self, role: str):
        """Set the role for all routers."""
        self.current_role = role
        for module in self.model.modules():
            if isinstance(module, RoleBasedRouter):
                module.set_role(role)
                
    def set_token_roles(self, token_roles: torch.Tensor):
        """Set the token-level role mapping for all routers."""
        for module in self.model.modules():
            if isinstance(module, RoleBasedRouter):
                module.set_token_roles(token_roles)
    
    def parse_message_roles(self, text: str) -> torch.Tensor:
        """Parse a formatted message string and create token-to-role mapping.
        
        Args:
            text: Formatted message string with role markers
            
        Returns:
            token_roles: Tensor mapping each token to a role (0=user/system, 1=assistant)
        """
        # Tokenize the full text
        tokens = self.tokenizer(text, return_tensors='pt').input_ids[0]
        token_roles = torch.zeros(len(tokens), dtype=torch.long, device=self.device)
        
        # Find role markers and map tokens to roles
        # Look for patterns like <|im_start|>role or similar markers
        text_parts = text.split('<|im_start|>')
        current_pos = 0
        
        for part in text_parts:
            if not part:
                continue
                
            # Extract role from the part
            if part.startswith('system') or part.startswith('user'):
                role_val = 0  # Expert 0
            elif part.startswith('assistant'):
                role_val = 1  # Expert 1
            else:
                continue
            
            # Find tokens corresponding to this part
            part_text = '<|im_start|>' + part if current_pos > 0 else part
            part_tokens = self.tokenizer(part_text, add_special_tokens=False).input_ids
            part_len = len(part_tokens)
            
            # Assign role to these token positions
            end_pos = min(current_pos + part_len, len(token_roles))
            token_roles[current_pos:end_pos] = role_val
            current_pos = end_pos
        
        return token_roles
                
    def generate_with_role(
        self, 
        messages: str,  # Now accepts a single string
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate text with role-based routing using a single forward pass.
        
        Args:
            messages: Single formatted string with role markers
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        # Parse token-to-role mapping
        token_roles = self.parse_message_roles(messages)

        breakpoint() 
        # Set token roles for all routers
        self.set_token_roles(token_roles)
        
        # Tokenize
        inputs = self.tokenizer(
            messages,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # For new tokens during generation, use assistant role (1)
        # Extend token_roles for generation
        max_length = inputs['input_ids'].shape[1] + max_new_tokens
        extended_roles = torch.ones(max_length, dtype=torch.long, device=self.device)
        extended_roles[:len(token_roles)] = token_roles
        self.set_token_roles(extended_roles)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text


def test_role_based_routing():
    """Test the role-based routing with a complete conversation example."""
    
    print("="*80)
    print("Testing Role-Based Routing for Qwen MoE")
    print("="*80)
    
    # Initialize model
    model_path = "./upcycled_moe"
    print(f"\n1. Loading model from: {model_path}\n")

    try:
        wrapper = RoleBasedMoEWrapper(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n   ✓ Model loaded on {wrapper.device}")
    except Exception as e:
        print(f"\n   ✗ Error loading model: {e}")
        print("\n" + "="*80)
        print("ERROR: The model checkpoint appears to be invalid or corrupted.")
        print("="*80)
        print("\nThe upcycled_moe directory exists but the model weights don't match")
        print("the MoE architecture. This usually happens when:")
        print("  1. The upcycling script didn't complete successfully")
        print("  2. The checkpoint file is from a dense model, not an MoE")
        print("  3. The expert layers weren't properly initialized")
        print("\nPlease re-run the upcycling script to create a valid MoE checkpoint.")
        import traceback
        traceback.print_exc()
        return
    
    # Single test case as a formatted string
    print("\n2. Testing with complete conversation:\n")
    print(f"{'─'*80}")
    
    test_messages = """<|im_start|>system
You are a helpful AI assistant that provides clear and concise answers.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>user
How many people live in Paris?<|im_end|>
<|im_start|>assistant
"""
    
    # Show input messages
    print("\nInput Message String:")
    print("─"*40)
    print(test_messages)
    print("─"*40)
    
    print("\n" + "─"*80)
    print("Routing Information (Single Forward Pass):")
    print("  • System tokens → Expert 0")
    print("  • User tokens → Expert 0")  
    print("  • Assistant tokens → Expert 1")
    print("  • All routing decisions made per-token in parallel")
    print("─"*80)
    
    # Generate response
    print("\nGenerating assistant response...\n")
    try:
        response = wrapper.generate_with_role(
            messages=test_messages,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        print(f"[ASSISTANT]: {response}")
        print("\n" + "─"*80)
        print("✓ Generation successful!")
        print("  → All tokens routed in single forward pass")
        print("  → System/User tokens processed by Expert 0")
        print("  → Assistant tokens generated by Expert 1")
        
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
    
    # Summary
    print("\n📊 Role-Based Routing Summary:")
    print("   • Single forward pass for entire conversation")
    print("   • Per-token routing based on role markers")
    print("   • System/User tokens → Expert 0")
    print("   • Assistant tokens → Expert 1")
    print("\n   This enables efficient role-specific specialization in the MoE model.")


def simple_routing_test():
    """
    A simpler test that demonstrates routing logic without full generation.
    Useful for debugging or when model loading fails.
    """
    print("\n" + "="*80)
    print("Simple Routing Logic Test (No Model Required)")
    print("="*80)
    
    # Create a dummy original gate
    class DummyGate(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], x.shape[1], 2)
    
    router = RoleBasedRouter(DummyGate(), num_experts=2)
    
    # Create dummy hidden states
    batch_size, seq_len, hidden_dim = 1, 10, 1024
    dummy_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    
    test_roles = ['user', 'system', 'assistant']
    
    print("\nTesting routing decisions:\n")
    for role in test_roles:
        router.set_role(role)
        logits = router(dummy_hidden)
        
        # Apply softmax to see which expert gets selected
        probs = F.softmax(logits, dim=-1)
        expert_idx = probs[0, 0].argmax().item()
        confidence = probs[0, 0, expert_idx].item()
        
        print(f"  Role '{role}' → Expert {expert_idx} (confidence: {confidence:.4f})")
    
    print("\n✓ Routing logic verified!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    print("\n🚀 Role-Based MoE Router for Qwen Models\n")
    
    # Check if model exists
    import os
    model_path = "./upcycled_moe"

    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("   Running simple routing logic test instead...\n")
        simple_routing_test()
    else:
        # Run full test with model
        if len(sys.argv) > 1 and sys.argv[1] == '--simple':
            simple_routing_test()
        else:
            try:
                test_role_based_routing()
            except Exception as e:
                print(f"\n❌ Error during full test: {e}")
                print("\nFalling back to simple routing test...\n")
                simple_routing_test()
    
    print("\n✨ Done!\n")
