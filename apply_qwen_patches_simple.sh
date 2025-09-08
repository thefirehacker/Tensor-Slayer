#!/bin/bash

# Qwen Model Patcher Script (Simple Version)
# This script applies the exact same patches that were detected in the modified Qwen model
# Place this script in the base model directory and run it to recreate the modifications

set -e  # Exit on any error

# Check if we're in the right directory
if [ ! -f "model.safetensors" ]; then
    echo "Error: model.safetensors not found in current directory"
    echo "Please run this script from the base model directory containing model.safetensors"
    exit 1
fi

# Use the parent directory's virtual environment if available
if [ -d "../venv" ]; then
    echo "Using parent directory virtual environment..."
    source ../venv/bin/activate
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "Using local virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
else
    echo "Using system python..."
    PYTHON_CMD="python3"
fi

echo "Creating patch application script..."

# Create the patching script
cat > apply_patches.py << 'EOF'
#!/usr/bin/env python3
"""
Apply Qwen Model Patches
Recreates the modifications found in the modified Qwen model
"""

import torch
import os
import sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import shutil
from datetime import datetime

def load_all_tensors(model_path):
    """Load all tensors from a safetensors file"""
    tensors = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def apply_scale_patch(tensor, factor):
    """Apply scaling to a tensor"""
    return tensor * factor

def apply_clamp_patch(tensor, min_val, max_val):
    """Apply clamping to a tensor"""
    return torch.clamp(tensor, min=min_val, max=max_val)

def main():
    model_path = "model.safetensors"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found in current directory")
        return 1
    
    print("Loading original model...")
    tensors = load_all_tensors(model_path)
    
    print(f"Loaded {len(tensors)} tensors")
    
    # Create backup
    backup_path = f"model_original_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.safetensors"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(model_path, backup_path)
    
    print("Applying patches...")
    
    # Apply scale operations (41 patches)
    scale_patches = [
        ("lm_head.weight", 1.03),
        ("model.embed_tokens.weight", 1.02),
        ("model.layers.0.input_layernorm.weight", 1.05),
        ("model.layers.0.mlp.gate_proj.weight", 1.05),
        ("model.layers.10.mlp.down_proj.weight", 1.02),
        ("model.layers.10.self_attn.q_proj.weight", 1.02),
        ("model.layers.11.mlp.down_proj.weight", 1.02),
        ("model.layers.11.self_attn.q_proj.weight", 1.02),
        ("model.layers.12.mlp.down_proj.weight", 1.02),
        ("model.layers.12.self_attn.q_proj.weight", 1.02),
        ("model.layers.13.mlp.down_proj.weight", 1.02),
        ("model.layers.13.self_attn.q_proj.weight", 1.02),
        ("model.layers.14.mlp.down_proj.weight", 1.02),
        ("model.layers.14.self_attn.q_proj.weight", 1.02),
        ("model.layers.15.mlp.down_proj.weight", 1.02),
        ("model.layers.15.mlp.up_proj.weight", 1.03),
        ("model.layers.15.self_attn.q_proj.weight", 1.02),
        ("model.layers.16.mlp.down_proj.weight", 1.02),
        ("model.layers.16.self_attn.q_proj.weight", 1.02),
        ("model.layers.17.mlp.down_proj.weight", 1.02),
        ("model.layers.17.self_attn.q_proj.weight", 1.02),
        ("model.layers.18.mlp.down_proj.weight", 1.02),
        ("model.layers.18.self_attn.q_proj.weight", 1.02),
        ("model.layers.19.mlp.down_proj.weight", 1.02),
        ("model.layers.19.self_attn.q_proj.weight", 1.02),
        ("model.layers.20.mlp.down_proj.weight", 1.02),
        ("model.layers.20.self_attn.q_proj.weight", 1.02),
        ("model.layers.21.mlp.down_proj.weight", 1.02),
        ("model.layers.21.self_attn.q_proj.weight", 1.02),
        ("model.layers.22.mlp.down_proj.weight", 1.02),
        ("model.layers.22.self_attn.q_proj.weight", 1.02),
        ("model.layers.23.mlp.down_proj.weight", 1.02),
        ("model.layers.23.self_attn.q_proj.weight", 1.02),
        ("model.layers.24.mlp.down_proj.weight", 1.02),
        ("model.layers.24.self_attn.q_proj.weight", 1.02),
        ("model.layers.25.mlp.down_proj.weight", 1.02),
        ("model.layers.25.self_attn.q_proj.weight", 1.02),
        ("model.layers.26.mlp.down_proj.weight", 1.02),
        ("model.layers.26.self_attn.q_proj.weight", 1.02),
        ("model.layers.27.mlp.down_proj.weight", 1.02),
        ("model.layers.27.self_attn.q_proj.weight", 1.02),
    ]
    
    # Apply clamp operations (3 patches)
    clamp_patches = [
        ("model.layers.15.self_attn.k_norm.weight", -0.0032958984375, 20.0),
        ("model.layers.27.input_layernorm.weight", 2.4375, 40.0),
        ("model.layers.27.post_attention_layernorm.weight", -0.0174560546875, 100.0),
    ]
    
    patches_applied = 0
    
    # Apply scale patches
    for tensor_name, factor in scale_patches:
        if tensor_name in tensors:
            print(f"Scaling {tensor_name} by factor {factor}")
            tensors[tensor_name] = apply_scale_patch(tensors[tensor_name], factor)
            patches_applied += 1
        else:
            print(f"Warning: Tensor {tensor_name} not found, skipping...")
    
    # Apply clamp patches
    for tensor_name, min_val, max_val in clamp_patches:
        if tensor_name in tensors:
            print(f"Clamping {tensor_name} to range [{min_val}, {max_val}]")
            tensors[tensor_name] = apply_clamp_patch(tensors[tensor_name], min_val, max_val)
            patches_applied += 1
        else:
            print(f"Warning: Tensor {tensor_name} not found, skipping...")
    
    # Save the modified model
    output_path = "model_patched.safetensors"
    print(f"Saving modified model to {output_path}...")
    save_file(tensors, output_path)
    
    print(f"Successfully applied {patches_applied} patches!")
    print(f"Original model backed up to: {backup_path}")
    print(f"Modified model saved as: {output_path}")
    print("")
    print("To use the modified model:")
    print(f"  mv {output_path} model.safetensors")
    print("")
    print("To restore original:")
    print(f"  mv {backup_path} model.safetensors")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

echo "Running patch application..."
$PYTHON_CMD apply_patches.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Patch application completed successfully!"
    echo ""
    echo "Files in directory:"
    ls -la *.safetensors
    echo ""
    echo "To compare with the reference modified model, run:"
    echo "  cd .. && source venv/bin/activate && python safetensors_diff_analyzer.py compare Qwen_0.6B/model_patched.safetensors Qwen_0.6B_modified/model.safetensors"
else
    echo "❌ Patch application failed!"
fi

# Clean up the Python script
rm -f apply_patches.py