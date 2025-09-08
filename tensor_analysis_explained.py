#!/usr/bin/env python3
"""
DETAILED EXPLANATION: Step 1 - Tensor Analysis
Shows exactly what happens when analyzing a tensor
"""

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def explain_tensor_analysis():
    """Step-by-step explanation of tensor analysis"""
    
    console.print("[bold cyan]🔍 STEP 1: TENSOR ANALYSIS - DETAILED BREAKDOWN[/]")
    console.print()
    
    # ===== EXAMPLE 1: Create a Sample Tensor =====
    console.print("[bold yellow]📊 Example 1: Attention Weight Tensor[/]")
    
    # This simulates what an actual attention weight tensor looks like
    torch.manual_seed(42)  # For reproducible results
    attention_tensor = torch.randn(896, 896) * 0.02  # 896x896 matrix, small values
    
    console.print(f"[cyan]Tensor Name:[/] model.layers.5.self_attn.q_proj.weight")
    console.print(f"[cyan]What it does:[/] Controls how words pay attention to each other")
    console.print()
    
    # ===== STEP-BY-STEP ANALYSIS =====
    
    # 1. SHAPE ANALYSIS
    console.print("[bold green]1. SHAPE ANALYSIS[/]")
    shape = attention_tensor.shape
    total_elements = attention_tensor.numel()
    console.print(f"   Shape: {shape}")
    console.print(f"   Meaning: {shape[0]} input features → {shape[1]} output features")
    console.print(f"   Total numbers in tensor: {total_elements:,}")
    console.print()
    
    # 2. VALUE DISTRIBUTION ANALYSIS
    console.print("[bold green]2. VALUE DISTRIBUTION ANALYSIS[/]")
    
    min_val = float(attention_tensor.min())
    max_val = float(attention_tensor.max())
    mean_val = float(attention_tensor.mean())
    std_val = float(attention_tensor.std())
    
    console.print(f"   Minimum value: {min_val:.6f}")
    console.print(f"   Maximum value: {max_val:.6f}")
    console.print(f"   Mean (average): {mean_val:.6f}")
    console.print(f"   Standard deviation: {std_val:.6f}")
    
    # Explain what these numbers mean
    console.print("\n   [bold]What this tells us:[/]")
    if abs(mean_val) < 0.001:
        console.print("   ✅ Mean near zero - Good! Tensor is balanced")
    else:
        console.print("   ⚠️  Mean not near zero - Tensor might be biased")
    
    if std_val < 0.01:
        console.print("   ⚠️  Low std - Values are too similar, low diversity")
    elif std_val > 0.1:
        console.print("   ⚠️  High std - Values vary too much, might be unstable")
    else:
        console.print("   ✅ Good std - Healthy value distribution")
    console.print()
    
    # 3. SPARSITY ANALYSIS
    console.print("[bold green]3. SPARSITY ANALYSIS[/]")
    zeros_count = (attention_tensor == 0).sum()
    zeros_percent = float(zeros_count / total_elements * 100)
    
    console.print(f"   Zero values: {zeros_count:,} out of {total_elements:,}")
    console.print(f"   Sparsity: {zeros_percent:.2f}%")
    
    if zeros_percent > 50:
        console.print("   ⚠️  High sparsity - Too many zeros, information loss")
    elif zeros_percent < 1:
        console.print("   ✅ Low sparsity - Dense tensor, rich information")
    else:
        console.print("   ✅ Moderate sparsity - Balanced")
    console.print()
    
    # 4. VALUE RANGE ANALYSIS
    console.print("[bold green]4. VALUE RANGE ANALYSIS[/]")
    value_range = max_val - min_val
    console.print(f"   Value range: {value_range:.6f} (from {min_val:.6f} to {max_val:.6f})")
    
    # Check for common problems
    if value_range < 0.001:
        console.print("   ⚠️  Very narrow range - Limited expressiveness")
        recommendation = "Consider scaling up (multiply by 1.05-1.1)"
    elif value_range > 2.0:
        console.print("   ⚠️  Very wide range - Might cause instability")
        recommendation = "Consider clamping values or normalizing"
    else:
        console.print("   ✅ Good range - Balanced expressiveness")
        recommendation = "Fine-tune with small scaling (1.02-1.05)"
    
    console.print(f"   [bold]Recommendation:[/] {recommendation}")
    console.print()
    
    # 5. PERCENTILE ANALYSIS
    console.print("[bold green]5. PERCENTILE ANALYSIS[/]")
    flat_tensor = attention_tensor.flatten()
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Percentile", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Meaning", style="yellow")
    
    for p in percentiles:
        value = float(torch.quantile(flat_tensor, p/100))
        if p == 50:
            meaning = "Median (middle value)"
        elif p <= 10:
            meaning = "Bottom values"
        elif p >= 90:
            meaning = "Top values"
        else:
            meaning = "Middle range"
        
        table.add_row(f"{p}%", f"{value:.6f}", meaning)
    
    console.print(table)
    console.print()
    
    # 6. PROBLEM DETECTION
    console.print("[bold green]6. AUTOMATIC PROBLEM DETECTION[/]")
    
    problems = []
    solutions = []
    
    # Check for common issues
    if abs(mean_val) > 0.01:
        problems.append("Mean too far from zero")
        solutions.append("Add small value to center: add -0.001")
    
    if std_val < 0.005:
        problems.append("Values too similar (low diversity)")
        solutions.append("Scale to increase diversity: scale 1.1")
    
    if zeros_percent > 30:
        problems.append("Too many zero values")
        solutions.append("Add small values to zeros: add 0.001")
    
    if value_range < 0.01:
        problems.append("Value range too narrow")
        solutions.append("Scale to widen range: scale 1.05")
    
    if problems:
        console.print("   [bold red]Problems detected:[/]")
        for i, (problem, solution) in enumerate(zip(problems, solutions), 1):
            console.print(f"   {i}. {problem}")
            console.print(f"      → Solution: {solution}")
    else:
        console.print("   [bold green]✅ No major problems detected![/]")
    console.print()
    
    # 7. FINAL ANALYSIS SUMMARY
    console.print(Panel(
        f"""[bold]TENSOR ANALYSIS SUMMARY[/]
        
[cyan]Tensor:[/] model.layers.5.self_attn.q_proj.weight
[cyan]Shape:[/] {shape}
[cyan]Total Elements:[/] {total_elements:,}

[yellow]Key Statistics:[/]
• Mean: {mean_val:.6f}
• Std: {std_val:.6f}  
• Range: {min_val:.6f} to {max_val:.6f}
• Sparsity: {zeros_percent:.2f}%

[green]Recommended Action:[/]
{recommendation}
        """,
        title="Analysis Complete",
        border_style="green"
    ))

def compare_tensors():
    """Show how different tensors have different characteristics"""
    
    console.print("\n[bold cyan]🔄 COMPARISON: Different Tensor Types[/]")
    console.print()
    
    # Create different types of tensors
    torch.manual_seed(42)
    
    # 1. Healthy tensor
    healthy = torch.randn(100, 100) * 0.02
    
    # 2. Problematic tensor (too small values)
    weak = torch.randn(100, 100) * 0.001  # Very small
    
    # 3. Problematic tensor (too many zeros)
    sparse = torch.randn(100, 100) * 0.02
    sparse[sparse.abs() < 0.01] = 0  # Zero out small values
    
    tensors = [
        ("Healthy Tensor", healthy),
        ("Weak Tensor", weak), 
        ("Sparse Tensor", sparse)
    ]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tensor Type", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="green") 
    table.add_column("Sparsity", style="yellow")
    table.add_column("Recommendation", style="red")
    
    for name, tensor in tensors:
        mean_val = float(tensor.mean())
        std_val = float(tensor.std())
        zeros_pct = float((tensor == 0).sum() / tensor.numel() * 100)
        
        # Generate recommendation
        if "Weak" in name:
            rec = "Scale by 1.1 (strengthen)"
        elif "Sparse" in name:
            rec = "Add 0.001 (fill zeros)"
        else:
            rec = "Scale by 1.02 (fine-tune)"
        
        table.add_row(
            name,
            f"{mean_val:.6f}",
            f"{std_val:.6f}",
            f"{zeros_pct:.1f}%",
            rec
        )
    
    console.print(table)
    console.print()
    
    console.print("[bold green]Key Insight:[/] Different problems need different solutions!")

if __name__ == "__main__":
    explain_tensor_analysis()
    compare_tensors()
