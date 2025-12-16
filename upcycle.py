#!/usr/bin/env python3
"""
TL;DR the script for upcycling works; if you want the exact same initial output as the dense model you need to set
num_experts_per_tok equal to n_experts, or modify the line `current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]`
https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L258

Script to upcycle a Qwen3 model to Qwen3 MoE model.
Transforms standard MLP layers into Mixture of Experts layers.
Simplified version focusing on Layer 0 analysis only.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file  # Add this import



def create_moe_config(base_config, n_experts: int = 2, num_experts_per_tok: int = 1):
    """
    Create MoE configuration from base Qwen3 config.
    
    Args:
        base_config: Base Qwen3 configuration
        n_experts: Number of experts to createco
        num_experts_per_tok: Number of experts activated per token
    
    Returns:
        Modified configuration for MoE model
    """
    config_dict = base_config.to_dict()
    
    # Update architecture to Qwen3MoeForCausalLM
    config_dict['architectures'] = ['Qwen3MoeForCausalLM']
    config_dict['model_type'] = 'qwen3_moe'
    
    # Add MoE specific parameters
    config_dict['num_experts'] = n_experts
    config_dict['num_experts_per_tok'] = num_experts_per_tok
    config_dict['router_aux_loss_coef'] = 0.001  # Default routing loss coefficient
    config_dict['output_router_logits'] = False
    
    # Expert configuration
    config_dict['moe_intermediate_size'] = config_dict.get('intermediate_size', 
                                                            int(config_dict['hidden_size'] * 8/3))
    config_dict['shared_expert_intermediate_size'] = 0  # No shared expert by default
    config_dict['norm_topk_prob'] = False
    
    return config_dict


def upcycle_mlp_to_moe(model, n_experts: int = 2, init_strategy: str = 'uniform'):
    """
    Transform MLP layers into MoE layers by duplicating weights.
    
    Args:
        model: Base Qwen3 model
        n_experts: Number of experts to create
        init_strategy: Strategy for initializing gate weights ('uniform', 'random', 'first_expert')
    
    Returns:
        State dict for MoE model
    """
    state_dict = model.state_dict()
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Handle embedding and final layers (unchanged)
        if 'embed_tokens' in key or 'lm_head' in key or 'norm' in key:
            new_state_dict[key] = value
            continue
            
        # Handle attention layers (unchanged)
        if 'self_attn' in key:
            new_state_dict[key] = value
            continue
            
        # Handle input layer norm (unchanged)
        if 'input_layernorm' in key:
            new_state_dict[key] = value
            continue
            
        # Handle post attention layer norm
        if 'post_attention_layernorm' in key:
            new_state_dict[key] = value
            continue
            
        # Transform MLP layers into expert layers
        if 'mlp' in key:
            # Extract layer index
            parts = key.split('.')
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = parts[i + 1]
                    break
                    
            if layer_idx is not None:
                # Create gate weights with different initialization strategies
                if 'gate_proj' in key and not any(f'layers.{layer_idx}.mlp.gate' in k for k in new_state_dict):
                    hidden_size = model.config.hidden_size
                    
                    if init_strategy == 'uniform':
                        # Initialize with small random values for uniform routing
                        # Not all zeros which would cause issues
                        gate_weight = torch.randn(n_experts, hidden_size) * 0.001
                    elif init_strategy == 'first_expert':
                        # Initialize to route mostly to first expert (mimics original behavior)
                        gate_weight = torch.zeros(n_experts, hidden_size)
                        gate_weight[0, :] = 1e9  # Strong bias to first expert
                        # Add small noise to other experts
                        if n_experts > 1:
                            gate_weight[1:, :] = 0 # torch.randn(n_experts-1, hidden_size) * 0.001
                    else:  # 'random'
                        gate_weight = torch.randn(n_experts, hidden_size) * 0.02
                    
                    gate_key = f'model.layers.{layer_idx}.mlp.gate.weight'
                    new_state_dict[gate_key] = gate_weight
                
                
                # self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
                # self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
                
                # Duplicate MLP weights for each expert
                for expert_idx in range(n_experts):
                    if 'gate_proj' in key:
                        expert_key = key.replace('mlp.gate_proj', f'mlp.experts.{expert_idx}.gate_proj')
                    elif 'up_proj' in key:
                        expert_key = key.replace('mlp.up_proj', f'mlp.experts.{expert_idx}.up_proj')
                    elif 'down_proj' in key:
                        expert_key = key.replace('mlp.down_proj', f'mlp.experts.{expert_idx}.down_proj')
                    else:
                        continue
                    
                    # Clone the weight for each expert
                    new_state_dict[expert_key] = value.clone()
        else:
            # Keep other weights unchanged
            new_state_dict[key] = value
    
    return new_state_dict


def save_moe_model(output_dir: Path, config_dict: dict, state_dict: dict, tokenizer):
    """
    Save the MoE model to disk in safetensors format.
    
    Args:
        output_dir: Directory to save the model
        config_dict: Model configuration dictionary
        state_dict: Model state dictionary
        tokenizer: Tokenizer to save
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Handle shared tensors (tied embeddings) by cloning them
    # This avoids the "tensors share memory" error
    state_dict_to_save = {}
    for key, tensor in state_dict.items():
        state_dict_to_save[key] = tensor.clone().contiguous()
    
    # Save model weights in safetensors format (required for SafetensorLazyLoader)
    save_file(state_dict_to_save, output_dir / 'model.safetensors')
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ MoE model saved to {output_dir}")


def track_layer0_divergence(original_model, moe_model, tokenizer, args, prompt: str = "What is the capital of France?"):
    """
    Track where hidden states diverge in Layer 0 only.
    
    Args:
        original_model: Original Qwen3 model
        moe_model: Upcycled MoE model
        tokenizer: Tokenizer for both models
        prompt: Test prompt
    
    Returns:
        Dictionary with layer 0 divergence metrics
    """
    from collections import OrderedDict
    
    print("\n" + "="*60)
    print("🔍 LAYER 0 DIVERGENCE ANALYSIS")
    print("="*60)
    
    device = next(original_model.parameters()).device
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Storage for hidden states - use OrderedDict to maintain order
    original_states = OrderedDict()
    moe_states = OrderedDict()
    
    # Define hook functions
    def make_io_hook(storage_dict, name):
        def hook(module, input, output):
            # Store input
            if isinstance(input, tuple) and len(input) > 0:
                hidden_state_in = input[0]
                if torch.is_tensor(hidden_state_in):
                    storage_dict[name + '_input'] = hidden_state_in.detach().clone()
            elif torch.is_tensor(input):
                storage_dict[name + '_input'] = input.detach().clone()
            
            # Store output
            if isinstance(output, tuple) and len(output) > 0:
                hidden_state_out = output[0]
                if torch.is_tensor(hidden_state_out):
                    storage_dict[name + '_output'] = hidden_state_out.detach().clone()
            elif torch.is_tensor(output):
                storage_dict[name + '_output'] = output.detach().clone()
        return hook
    
    # Special hook for activation functions
    def make_activation_hook(storage_dict, name):
        def hook(module, input, output):
            if torch.is_tensor(input[0]):
                storage_dict[name + '_input'] = input[0].detach().clone()
            if torch.is_tensor(output):
                storage_dict[name + '_output'] = output.detach().clone()
        return hook
    
    # Register hooks for LAYER 0 ONLY in original model
    original_hooks = []
    
    layer_0 = original_model.model.layers[0]
    
    # Layer 0 overall input/output
    hook = layer_0.register_forward_hook(
        make_io_hook(original_states, 'layer_00')
    )
    original_hooks.append(hook)
    
    # Layer 0 self-attention
    hook = layer_0.self_attn.register_forward_hook(
        make_io_hook(original_states, 'layer_00_attn')
    )
    original_hooks.append(hook)
    
    # Layer 0 MLP overall
    hook = layer_0.mlp.register_forward_hook(
        make_io_hook(original_states, 'layer_00_mlp')
    )
    original_hooks.append(hook)
    
    # Layer 0 MLP components
    if hasattr(layer_0.mlp, 'gate_proj'):
        hook = layer_0.mlp.gate_proj.register_forward_hook(
            make_io_hook(original_states, 'layer_00_mlp_gate_proj')
        )
        original_hooks.append(hook)
    
    if hasattr(layer_0.mlp, 'up_proj'):
        hook = layer_0.mlp.up_proj.register_forward_hook(
            make_io_hook(original_states, 'layer_00_mlp_up_proj')
        )
        original_hooks.append(hook)
    
    if hasattr(layer_0.mlp, 'act_fn'):
        hook = layer_0.mlp.act_fn.register_forward_hook(
            make_activation_hook(original_states, 'layer_00_mlp_act_fn')
        )
        original_hooks.append(hook)
    
    if hasattr(layer_0.mlp, 'down_proj'):
        hook = layer_0.mlp.down_proj.register_forward_hook(
            make_io_hook(original_states, 'layer_00_mlp_down_proj')
        )
        original_hooks.append(hook)
    
    # Register hooks for LAYER 0 ONLY in MoE model
    moe_hooks = []
    
    moe_layer_0 = moe_model.model.layers[0]
    
    # Layer 0 overall input/output
    hook = moe_layer_0.register_forward_hook(
        make_io_hook(moe_states, 'layer_00')
    )
    moe_hooks.append(hook)
    
    # Layer 0 self-attention
    hook = moe_layer_0.self_attn.register_forward_hook(
        make_io_hook(moe_states, 'layer_00_attn')
    )
    moe_hooks.append(hook)
    
    # Layer 0 MLP/MoE overall
    hook = moe_layer_0.mlp.register_forward_hook(
        make_io_hook(moe_states, 'layer_00_mlp')
    )
    moe_hooks.append(hook)
    
    # Layer 0 Gate/Router
    if hasattr(moe_layer_0.mlp, 'gate'):
        hook = moe_layer_0.mlp.gate.register_forward_hook(
            make_io_hook(moe_states, 'layer_00_mlp_gate')
        )
        moe_hooks.append(hook)
    
    # Layer 0 Expert 0 components
    if hasattr(moe_layer_0.mlp, 'experts') and len(moe_layer_0.mlp.experts) > 0:
        for expert_id, expert_i in enumerate(moe_layer_0.mlp.experts):
            
            if hasattr(expert_i, 'gate_proj'):
                hook = expert_i.gate_proj.register_forward_hook(
                    make_io_hook(moe_states, f'layer_00_mlp_expert{expert_id}_gate_proj')
                )
                moe_hooks.append(hook)
            
            if hasattr(expert_i, 'up_proj'):
                hook = expert_i.up_proj.register_forward_hook(
                    make_io_hook(moe_states, f'layer_00_mlp_expert{expert_id}_up_proj')
                )
                moe_hooks.append(hook)
            
            if hasattr(expert_i, 'act_fn'):
                hook = expert_i.act_fn.register_forward_hook(
                    make_activation_hook(moe_states, f'layer_00_mlp_expert{expert_id}_act_fn')
                )
                moe_hooks.append(hook)
            
            if hasattr(expert_i, 'down_proj'):
                hook = expert_i.down_proj.register_forward_hook(
                    make_io_hook(moe_states, f'layer_00_mlp_expert{expert_id}_down_proj')
                )
                moe_hooks.append(hook)
    
    # Forward pass through both models
    with torch.no_grad():
        orig_out = original_model(**inputs)
        moe_out = moe_model(**inputs)
    
    # Remove hooks
    for hook in original_hooks:
        hook.remove()
    for hook in moe_hooks:
        hook.remove()
    
    # Greedy sampling generation
    print("\n" + "="*60)
    print("📝 GREEDY GENERATION COMPARISON")
    print("="*60)
    
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate with original model using greedy decoding
    print("\n🔵 Original Model Output:")
    with torch.no_grad():
        original_outputs = original_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    print(f"   {original_text}")
    
    # Generate with MoE model using greedy decoding
    print("\n🟢 MoE Model Output:")
    with torch.no_grad():
        moe_outputs = moe_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    moe_text = tokenizer.decode(moe_outputs[0], skip_special_tokens=True)
    print(f"   {moe_text}")
    
    # Check if outputs match
    outputs_match = original_text == moe_text
    print(f"\n✓ Outputs match: {outputs_match}")
    if not outputs_match:
        print("   ⚠️ Outputs differ - this is expected due to MoE routing mechanism")
    
    # Analyze logits difference
    print("\n" + "="*60)
    print("📊 LOGITS COMPARISON")
    print("="*60)
    
    # Get logits for next token prediction
    orig_logits = orig_out.logits[0, -1, :]  # Last token logits
    moe_logits = moe_out.logits[0, -1, :]
    
    # Compare top predictions
    orig_top5 = torch.topk(orig_logits, 5)
    moe_top5 = torch.topk(moe_logits, 5)
    
    print("\nTop 5 predictions for next token:")
    print("Original Model:")
    for i, (idx, score) in enumerate(zip(orig_top5.indices, orig_top5.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' (score: {score.item():.2f})")
    
    print("\nMoE Model:")
    for i, (idx, score) in enumerate(zip(moe_top5.indices, moe_top5.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{token}' (score: {score.item():.2f})")
    
    # Compute logits MSE
    logits_mse = F.mse_loss(orig_logits, moe_logits).item()
    print(f"\nLogits MSE: {logits_mse:.6f}")
    
    # Layer 0 divergence analysis
    print("\n" + "="*60)
    print("📊 LAYER 0 COMPONENT DIVERGENCE")
    print("="*60)
    
    if args.num_experts_per_tok  == args.n_experts:
        # Compare MLP components
        import re
        for key in moe_states.keys():
            if 'gate_input' in key or 'gate_output' in key: 
                continue
            original_key = re.sub(r'_expert\d+', '', key)
            if original_key in original_states:
                l2_norm = F.mse_loss(
                    original_states[original_key], 
                    moe_states[key],
                    reduction='sum'
                ).item()
                print(f'Comparing {original_key} with {key}: L2 norm = {l2_norm:.6f}')
        
        # Analyze gate routing if available
        if 'layer_00_mlp_gate_output' in moe_states:
            gate_output = moe_states['layer_00_mlp_gate_output']
            print(f"\n🎛️ MoE Gate Routing Analysis:")
            print(f"   Raw gate outputs: {gate_output[0].tolist()}")
            
            # Apply softmax to get routing probabilities
            expert_probs = torch.softmax(gate_output, dim=-1)
            print(f"   Expert probabilities: {expert_probs[0].tolist()}")
            
            # Which experts are selected (top-k)
            topk_values, topk_indices = torch.topk(expert_probs[0], k=min(2, expert_probs.shape[-1]))
            print(f"   Selected expert(s): {topk_indices.tolist()} with weights {topk_values.tolist()}")
        
    return {}


def compare_models(original_model_id: str, moe_model_path: str, args, prompt: str = "What is the capital of France?"):
    """
    Compare outputs from original and MoE models.
    
    Args:
        original_model_id: Model ID of the original model
        moe_model_path: Path to the upcycled MoE model
        prompt: Test prompt to generate from
    """
    print("\n" + "="*60)
    print("🔬 MODEL COMPARISON")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load original model
    print(f"\n📥 Loading original model: {original_model_id}")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)
    
    # Load MoE model
    print(f"📥 Loading MoE model: {moe_model_path}")
    moe_model = AutoModelForCausalLM.from_pretrained(
        moe_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    moe_tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)

    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    breakpoint()



    # Set pad token if not set
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token
    if moe_tokenizer.pad_token is None:
        moe_tokenizer.pad_token = moe_tokenizer.eos_token
    
    # Track Layer 0 divergence
    divergence_metrics = track_layer0_divergence(original_model, moe_model, original_tokenizer, args, prompt)
    
    print("\n" + "="*60)
    print("💡 Analysis Summary:")
    print("="*60)
    
    # Check where divergence starts
    for component, metrics in divergence_metrics.items():
        if metrics['output_rel_error'] and metrics['output_rel_error'] > 1.0:
            print(f"\n⚠️ First significant divergence (>1%) at: {component}")
            print(f"   Output differs by {metrics['output_rel_error']:.2f}%")
            break
    
    print("\n📌 Note: Even with identical experts, the gate/router mechanism")
    print("   fundamentally changes the computation path through weighted averaging.")
    
    # Clean up
    del original_model
    del moe_model
    torch.cuda.empty_cache() if device == "cuda" else None


def main():
    parser = argparse.ArgumentParser(description='Upcycle Qwen3 model to Qwen3 MoE')
    parser.add_argument('--model_id', type=str, required=True,
                       help='Model ID or path (e.g., Qwen/Qwen3-0.6B)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for the MoE model')
    parser.add_argument('--n_experts', type=int, default=2,
                       help='Number of experts (default: 2)')
    parser.add_argument('--num_experts_per_tok', type=int, default=1,
                       help='Number of experts activated per token (default: 1)')
    parser.add_argument('--gate_init', type=str, default='uniform',
                       choices=['uniform', 'random', 'first_expert'],
                       help='Gate weight initialization strategy (default: uniform)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to load model on (default: cpu)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare original and MoE model outputs after upcycling')
    parser.add_argument('--test_prompt', type=str, default="What is the capital of France?",
                       help='Prompt to test models with (default: "What is the capital of France?")')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting upcycling of {args.model_id} to MoE with {args.n_experts} experts")
    print(f"   Gate initialization: {args.gate_init}")
    
    # Load the base model and tokenizer
    print("📥 Loading base model and tokenizer...")
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Verify it's a Qwen3 model
    if 'Qwen3ForCausalLM' not in config.architectures:
        raise ValueError(f"Model {args.model_id} is not a Qwen3 model. Found architectures: {config.architectures}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device != 'cpu' else torch.float32,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Create MoE configuration
    print("⚙️ Creating MoE configuration...")
    moe_config_dict = create_moe_config(
        config, 
        n_experts=args.n_experts,
        num_experts_per_tok=args.num_experts_per_tok
    )
    
    # Upcycle the model weights
    print(f"🔧 Upcycling MLP layers to {args.n_experts} experts...")
    moe_state_dict = upcycle_mlp_to_moe(model, n_experts=args.n_experts, init_strategy=args.gate_init)
    
    # Save the MoE model
    print("💾 Saving MoE model...")
    output_path = Path(args.output_dir)
    save_moe_model(output_path, moe_config_dict, moe_state_dict, tokenizer)
    
    # Print summary
    print("\n" + "="*50)
    print("✨ Upcycling complete!")
    print(f"📊 Model: {args.model_id} → {args.output_dir}")
    print(f"🧠 Experts: {args.n_experts} total, {args.num_experts_per_tok} active per token")
    print(f"🎛️ Gate init: {args.gate_init}")
    print(f"📁 Output saved to: {output_path.absolute()}")
    print("="*50)
    
    # Run comparison if requested
    if args.compare:
        compare_models(args.model_id, args.output_dir, args, args.test_prompt)


if __name__ == "__main__":
    main()