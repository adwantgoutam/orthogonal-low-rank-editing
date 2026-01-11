"""
Generate comparison plots for naive vs orthogonal editing.

Produces 6 plots:
1. Edit Success vs Number of Edits
2. Locality vs Number of Edits
3. Condition Number vs Number of Edits
4. Interference Index vs Number of Edits
5. CN vs Edit Success (scatter)
6. Interference vs Locality (scatter)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_results(json_path: str) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_metrics(results: Dict, mode: str) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Extract metrics for a given mode.
    
    Returns:
        (scales, edit_success, locality, condition_number, interference_index)
    """
    scales = []
    edit_success = []
    locality = []
    condition_number = []
    interference_index = []
    
    for key, data in results.items():
        if data.get('mode') == mode:
            # Get scale from num_edits or parse from key
            scale = data.get('num_edits', int(key.split('_')[0]) if '_' in key else int(key))
            scales.append(scale)
            edit_success.append(data.get('edit_success', 0.0))
            locality.append(data.get('locality', 0.0))
            condition_number.append(data.get('condition_number', 1.0))
            interference_index.append(data.get('interference_index', 0.0))
    
    # Sort by scale
    sorted_pairs = sorted(zip(scales, edit_success, locality, condition_number, interference_index))
    scales, edit_success, locality, condition_number, interference_index = zip(*sorted_pairs) if sorted_pairs else ([], [], [], [], [])
    
    return list(scales), list(edit_success), list(locality), list(condition_number), list(interference_index)

def plot_edit_success(scales_naive, success_naive, scales_ortho, success_ortho, output_dir: Path):
    """Plot 1: Edit Success vs Number of Edits"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plt.plot(scales_naive, success_naive, 'o-', label='Naive', linewidth=2, markersize=8, color='#d62728')
    if scales_ortho:
        plt.plot(scales_ortho, success_ortho, 's-', label='Orthogonal', linewidth=2, markersize=8, color='#2ca02c')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Edit Success Rate', fontsize=12)
    plt.title('Edit Success vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot1_edit_success.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot1_edit_success.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 1: Edit Success vs Number of Edits")

def plot_locality(scales_naive, locality_naive, scales_ortho, locality_ortho, output_dir: Path):
    """Plot 2: Locality vs Number of Edits"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plt.plot(scales_naive, locality_naive, 'o-', label='Naive', linewidth=2, markersize=8, color='#d62728')
    if scales_ortho:
        plt.plot(scales_ortho, locality_ortho, 's-', label='Orthogonal', linewidth=2, markersize=8, color='#2ca02c')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Locality Score', fontsize=12)
    plt.title('Locality Preservation vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot2_locality.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot2_locality.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 2: Locality vs Number of Edits")

def plot_condition_number(scales_naive, cn_naive, scales_ortho, cn_ortho, output_dir: Path):
    """Plot 3: Condition Number vs Number of Edits"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plt.plot(scales_naive, cn_naive, 'o-', label='Naive', linewidth=2, markersize=8, color='#d62728')
    if scales_ortho:
        plt.plot(scales_ortho, cn_ortho, 's-', label='Orthogonal', linewidth=2, markersize=8, color='#2ca02c')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Condition Number', fontsize=12)
    plt.title('Condition Number vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'plot3_condition_number.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot3_condition_number.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 3: Condition Number vs Number of Edits")

def plot_interference(scales_naive, ii_naive, scales_ortho, ii_ortho, output_dir: Path):
    """Plot 4: Interference Index vs Number of Edits"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plt.plot(scales_naive, ii_naive, 'o-', label='Naive', linewidth=2, markersize=8, color='#d62728')
    if scales_ortho:
        plt.plot(scales_ortho, ii_ortho, 's-', label='Orthogonal', linewidth=2, markersize=8, color='#2ca02c')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Interference Index', fontsize=12)
    plt.title('Interference Index vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot4_interference.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot4_interference.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 4: Interference Index vs Number of Edits")

def plot_cn_vs_success(cn_naive, success_naive, cn_ortho, success_ortho, output_dir: Path):
    """Plot 5: CN vs Edit Success (scatter)"""
    plt.figure(figsize=(8, 6))
    if cn_naive and success_naive:
        plt.scatter(cn_naive, success_naive, label='Naive', s=100, alpha=0.7, color='#d62728', edgecolors='black', linewidth=1.5)
    if cn_ortho and success_ortho:
        plt.scatter(cn_ortho, success_ortho, label='Orthogonal', s=100, alpha=0.7, color='#2ca02c', edgecolors='black', linewidth=1.5, marker='s')
    plt.xlabel('Condition Number', fontsize=12)
    plt.ylabel('Edit Success Rate', fontsize=12)
    plt.title('Condition Number vs Edit Success', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot5_cn_vs_success.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot5_cn_vs_success.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 5: CN vs Edit Success (scatter)")

def plot_interference_vs_locality(ii_naive, locality_naive, ii_ortho, locality_ortho, output_dir: Path):
    """Plot 6: Interference vs Locality (scatter)"""
    plt.figure(figsize=(8, 6))
    if ii_naive and locality_naive:
        plt.scatter(ii_naive, locality_naive, label='Naive', s=100, alpha=0.7, color='#d62728', edgecolors='black', linewidth=1.5)
    if ii_ortho and locality_ortho:
        plt.scatter(ii_ortho, locality_ortho, label='Orthogonal', s=100, alpha=0.7, color='#2ca02c', edgecolors='black', linewidth=1.5, marker='s')
    plt.xlabel('Interference Index', fontsize=12)
    plt.ylabel('Locality Score', fontsize=12)
    plt.title('Interference vs Locality Preservation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot6_interference_vs_locality.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot6_interference_vs_locality.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 6: Interference vs Locality (scatter)")

def generate_summary_table(results: Dict, output_dir: Path):
    """Generate a summary table for the paper."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of Naive vs Orthogonal Editing}")
    lines.append("\\label{tab:comparison}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Mode & Edits & Success & Locality & CN & Interference \\\\")
    lines.append("\\midrule")
    
    # Extract and sort by scale
    all_data = []
    for key, data in results.items():
        # Handle both "1" and "1_naive" formats
        if '_' in key:
            scale = int(key.split('_')[0])
        else:
            scale = int(key)
        all_data.append((scale, data))
    all_data.sort(key=lambda x: x[0])  # Sort by scale only
    
    for scale, data in all_data:
        mode = data.get('mode', 'unknown')
        success = data.get('edit_success', 0.0)
        locality = data.get('locality', 0.0)
        cn = data.get('condition_number', 1.0)
        ii = data.get('interference_index', 0.0)
        lines.append(f"{mode.capitalize()} & {scale} & {success:.3f} & {locality:.3f} & {cn:.3f} & {ii:.3f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_dir / 'summary_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    print("✓ Summary table generated: summary_table.tex")

def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--input", type=str, default="../data/hybrid_experimental_results_fixed.json",
                       help="Input JSON file with results")
    parser.add_argument("--output_dir", type=str, default="../data/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    
    # Extract metrics
    scales_naive, success_naive, locality_naive, cn_naive, ii_naive = extract_metrics(results, 'naive')
    scales_ortho, success_ortho, locality_ortho, cn_ortho, ii_ortho = extract_metrics(results, 'orthogonal')
    
    print(f"\nNaive mode: {len(scales_naive)} data points")
    print(f"Orthogonal mode: {len(scales_ortho)} data points")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_edit_success(scales_naive, success_naive, scales_ortho, success_ortho, output_dir)
    plot_locality(scales_naive, locality_naive, scales_ortho, locality_ortho, output_dir)
    plot_condition_number(scales_naive, cn_naive, scales_ortho, cn_ortho, output_dir)
    plot_interference(scales_naive, ii_naive, scales_ortho, ii_ortho, output_dir)
    plot_cn_vs_success(cn_naive, success_naive, cn_ortho, success_ortho, output_dir)
    plot_interference_vs_locality(ii_naive, locality_naive, ii_ortho, locality_ortho, output_dir)
    
    # Generate summary table
    generate_summary_table(results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ All plots generated in {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

