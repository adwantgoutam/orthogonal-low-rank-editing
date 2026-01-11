"""
Simple Rank-One Model Editing (ROME) Implementation

This module implements a simplified gradient-based Rank-1 update for knowledge editing.
It computes the update direction u = -grad(L) and input activation v = h
to minimize the loss on the target fact.

@author: gadwant
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer

def compute_u_v_gradient(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    subject: str,
    relation: str,
    target_new: str,
    layer_idx: int,
    layer_name: str = "mlp",
    device: str = "cuda",
    learning_rate: float = 1e-1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rank-1 update vectors (u, v) using value-gradient approximation.
    
    This approximates the ROME update W' = W + u v^T where:
    - v is the input activation 'h' at the target layer for the prompt.
    - u is the gradient of the loss w.r.t the layer output, scaled by learning rate.
      This moves the output in the direction that minimizes loss.
    
    Args:
        model: The model to edit
        tokenizer: Tokenizer
        subject: Subject entity
        relation: Relation string
        target_new: New target object
        layer_idx: Index of layer to edit
        layer_name: Name of layer component
        device: Device
        learning_rate: Effective step size for the rank-1 update
        
    Returns:
        (u, v) rank-1 vectors
    """
    model.eval()
    
    # Construct prompt
    prompt = f"{subject} {relation}"
    full_text = f"{prompt} {target_new}"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    targets = tokenizer(full_text, return_tensors="pt").to(device)
    
    # We need the last token index of the prompt to extract activations
    prompt_len = inputs.input_ids.shape[1]
    
    # Hook to capture input activation (v) and backward gradient (u)
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        # input is a tuple (tensor,)
        # clone and detach to save state
        activations['input'] = input[0].detach().clone()
        
    def backward_hook(module, grad_input, grad_output):
        # grad_output is a tuple (tensor,)
        # We want the gradient w.r.t the output of the linear layer
        gradients['output'] = grad_output[0].detach().clone()

    # Identify target layer
    target_module = None
    
    # Logic to find the specific linear layer module
    # Adapted from orthogonal_editing.py logic
    if hasattr(model, 'gpt_neox'):
        # Pythia / GPT-NeoX
        layer = model.gpt_neox.layers[layer_idx]
        if layer_name == "mlp":
            # For MLP, we usually edit the output projection (dense_4h_to_h)
            # This aligns with where ROME edits (MLP output)
            target_module = layer.mlp.dense_4h_to_h
    elif hasattr(model, 'transformer'):
         # GPT-2
        layer = model.transformer.h[layer_idx]
        if layer_name == "mlp":
            target_module = layer.mlp.c_proj
            
    if target_module is None:
        raise ValueError(f"Could not find target layer {layer_idx} {layer_name}")

    # Register hooks
    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)
    
    # 1. Forward pass to get v (input activation)
    # We run the prompt + target to get gradients, but v refers to the prompt context
    # We generally take the activation at the LAST token of the subject/prompt
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass with gradients enabled for the target module
    # We need to run the full text to compute loss on the target
    outputs = model(targets.input_ids, labels=targets.input_ids)
    loss = outputs.loss
    
    # 2. Backward pass to get u (gradient signal)
    loss.backward()
    
    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    # Extract vectors
    # v: Input activation at the last token of the prompt (before target generation starts)
    # Shape: [batch, seq_len, hidden_dim] -> [hidden_dim]
    # We select the last token of the prompt part
    # Note: inputs are [batch, seq], activation is [batch, seq, dim]
    
    # IMPORTANT: The hook captured activation for the whole sequence (prompt + target)
    # We want v from the prompt end position
    v_full = activations['input'] # [1, seq_len, in_dim]
    v = v_full[0, prompt_len-1, :].clone() # Vector at last prompt token
    
    # u: Gradient w.r.t output at the same position
    # The gradient tells us how we should move the output vector to minimize loss
    # grad_output is dL/d(Wx), strictly we want to move Wx in direction -dL/d(Wx)
    grad_full = gradients['output'] # [1, seq_len, out_dim]
    u_grad = grad_full[0, prompt_len-1, :].clone()
    
    # u_update = -learning_rate * u_grad
    # This is a crude approximation of (K^{-1} (v* - v)) in ROME, but valid for Rank-1
    u = -learning_rate * u_grad
    
    # Clean up
    del activations
    del gradients
    
    return u, v
