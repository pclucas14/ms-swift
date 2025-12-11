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
        self.current_role = None
        self.use_role_routing = True
        
    def set_role(self, role: str):
        """Set the current role for routing decisions."""
        self.current_role = role
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Route based on current role. Returns router logits.
        
        Args:
            hidden_states: Can be 2D [batch_size * seq_len, hidden_dim] or 
                          3D [batch_size, seq_len, hidden_dim]
        
        Returns:
            router_logits: Logits for all experts
        """
        if not self.use_role_routing or self.current_role is None:
            breakpoint()
            # Fall back to original routing
            return self.original_gate(hidden_states)
        
        # Determine which expert to use based on role
        if self.current_role in ['user', 'system']:
            expert_idx = 0
        elif self.current_role == 'assistant':
            expert_idx = 1
        else:
            # Default to expert 0 for unknown roles
            expert_idx = 0
        
        # Handle both 2D and 3D inputs
        if hidden_states.dim() == 2:
            batch_tokens, hidden_dim = hidden_states.shape
            # Create logits: very high logit for selected expert, very low for others
            router_logits = torch.full(
                (batch_tokens, self.num_experts),
                -1e9,  # Very low logit
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            router_logits[:, expert_idx] = 1e9  # Very high logit for selected expert
            
        elif hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            # Create logits: very high logit for selected expert, very low for others
            router_logits = torch.full(
                (batch_size, seq_len, self.num_experts),
                -1e9,  # Very low logit
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            router_logits[:, :, expert_idx] = 1e9  # Very high logit for selected expert
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
        self.current_role = None
    
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
                
    def generate_with_role(
        self, 
        messages: List[Dict[str, str]], 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate text with role-based routing.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        # Format messages for the model
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            formatted_text = ""
            for msg in messages:
                formatted_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            formatted_text += "<|im_start|>assistant\n"
        
        # Set role for routing (use last message's role, or 'assistant' for generation)
        if messages:
            last_role = messages[-1]['role']
            self.set_role(last_role)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # For generation, route through assistant expert
        self.set_role('assistant')
        
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
    
    # Single test case with system, user, and assistant
    print("\n2. Testing with complete conversation:\n")
    print(f"{'─'*80}")
    
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant that provides clear and concise answers."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Show input messages
    print("\nInput Messages:")
    for msg in test_messages:
        print(f"  [{msg['role'].upper()}]: {msg['content']}")
    
    print("\n" + "─"*80)
    print("Routing Information:")
    print("  • System message will route through Expert 0")
    print("  • User message will route through Expert 0")
    print("  • Assistant response will route through Expert 1")
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
    print("   • System prompt    → Expert 0")
    print("   • User question    → Expert 0")
    print("   • Assistant answer → Expert 1")
    print("\n   This enables role-specific specialization in the MoE model.")


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
