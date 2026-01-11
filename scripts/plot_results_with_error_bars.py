"""
Generate plots with error bars from aggregated results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_aggregated_results(json_path: str) -> Dict:
    """Load aggregated results with mean ± std."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_metrics_with_error_bars(results: Dict, mode: str) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Extract metrics with error bars for a given mode.
    
    Returns:
        (scales, success_mean, success_std, locality_mean, locality_std,
         cn_mean, cn_std, ii_mean, ii_std)
    """
    scales = []
    success_mean = []
    success_std = []
    locality_mean = []
    locality_std = []
    cn_mean = []
    cn_std = []
    ii_mean = []
    ii_std = []
    
    for key, data in results.items():
        if data.get('mode') == mode:
            scale = data.get('num_edits', int(key.split('_')[0]) if '_' in key else int(key))
            scales.append(scale)
            
            # Handle both aggregated (dict) and single-run (float) formats
            if isinstance(data.get('edit_success'), dict):
                success_mean.append(data['edit_success']['mean'])
                success_std.append(data['edit_success']['std'])
            else:
                success_mean.append(data.get('edit_success', 0.0))
                success_std.append(0.0)
            
            if isinstance(data.get('locality'), dict):
                locality_mean.append(data['locality']['mean'])
                locality_std.append(data['locality']['std'])
            else:
                locality_mean.append(data.get('locality', 0.0))
                locality_std.append(0.0)
            
            if isinstance(data.get('condition_number'), dict):
                cn_mean.append(data['condition_number']['mean'])
                cn_std.append(data['condition_number']['std'])
            else:
                cn_mean.append(data.get('condition_number', 1.0))
                cn_std.append(0.0)
            
            if isinstance(data.get('interference_index'), dict):
                ii_mean.append(data['interference_index']['mean'])
                ii_std.append(data['interference_index']['std'])
            else:
                ii_mean.append(data.get('interference_index', 0.0))
                ii_std.append(0.0)
    
    # Sort by scale
    sorted_pairs = sorted(zip(scales, success_mean, success_std, locality_mean, locality_std, cn_mean, cn_std, ii_mean, ii_std))
    if sorted_pairs:
        scales, success_mean, success_std, locality_mean, locality_std, cn_mean, cn_std, ii_mean, ii_std = zip(*sorted_pairs)
        return (list(scales), list(success_mean), list(success_std), list(locality_mean), list(locality_std),
                list(cn_mean), list(cn_std), list(ii_mean), list(ii_std))
    else:
        return ([], [], [], [], [], [], [], [], [])

def plot_with_error_bars(scales, means, stds, label, color, marker, ax=None):
    """Plot with error bars."""
    if ax is None:
        ax = plt.gca()
    ax.errorbar(scales, means, yerr=stds, label=label, color=color, marker=marker,
                linewidth=2, markersize=8, capsize=4, capthick=1.5, alpha=0.8)

def plot_edit_success_with_bars(scales_naive, success_naive, std_naive, scales_ortho, success_ortho, std_ortho, output_dir: Path):
    """Plot 1: Edit Success vs Number of Edits (with error bars)"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plot_with_error_bars(scales_naive, success_naive, std_naive, 'Naive', '#d62728', 'o')
    if scales_ortho:
        plot_with_error_bars(scales_ortho, success_ortho, std_ortho, 'Orthogonal', '#2ca02c', 's')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Edit Success Rate', fontsize=12)
    plt.title('Edit Success vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot1_edit_success_error_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot1_edit_success_error_bars.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 1: Edit Success vs Number of Edits (with error bars)")

def plot_condition_number_with_bars(scales_naive, cn_naive, std_naive, scales_ortho, cn_ortho, std_ortho, output_dir: Path):
    """Plot 3: Condition Number vs Number of Edits (with error bars)"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plot_with_error_bars(scales_naive, cn_naive, std_naive, 'Naive', '#d62728', 'o')
    if scales_ortho:
        plot_with_error_bars(scales_ortho, cn_ortho, std_ortho, 'Orthogonal', '#2ca02c', 's')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Condition Number', fontsize=12)
    plt.title('Condition Number vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'plot3_condition_number_error_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot3_condition_number_error_bars.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 3: Condition Number vs Number of Edits (with error bars)")

def plot_interference_with_bars(scales_naive, ii_naive, std_naive, scales_ortho, ii_ortho, std_ortho, output_dir: Path):
    """Plot 4: Interference Index vs Number of Edits (with error bars)"""
    plt.figure(figsize=(8, 6))
    if scales_naive:
        plot_with_error_bars(scales_naive, ii_naive, std_naive, 'Naive', '#d62728', 'o')
    if scales_ortho:
        plot_with_error_bars(scales_ortho, ii_ortho, std_ortho, 'Orthogonal', '#2ca02c', 's')
    plt.xlabel('Number of Edits', fontsize=12)
    plt.ylabel('Interference Index', fontsize=12)
    plt.title('Interference Index vs Number of Edits', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot4_interference_error_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plot4_interference_error_bars.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Plot 4: Interference Index vs Number of Edits (with error bars)")

def main():
    parser = argparse.ArgumentParser(description="Generate plots with error bars")
    parser.add_argument("--input", type=str, required=True,
                       help="Input aggregated JSON file")
    parser.add_argument("--output_dir", type=str, default="../data/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading aggregated results from {args.input}...")
    results = load_aggregated_results(args.input)
    
    # Extract metrics
    scales_naive, success_naive, std_naive, _, _, cn_naive, cn_std_naive, ii_naive, ii_std_naive = extract_metrics_with_error_bars(results, 'naive')
    scales_ortho, success_ortho, std_ortho, _, _, cn_ortho, cn_std_ortho, ii_ortho, ii_std_ortho = extract_metrics_with_error_bars(results, 'orthogonal')
    
    print(f"\nNaive mode: {len(scales_naive)} data points")
    print(f"Orthogonal mode: {len(scales_ortho)} data points")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots with error bars...")
    plot_edit_success_with_bars(scales_naive, success_naive, std_naive, scales_ortho, success_ortho, std_ortho, output_dir)
    plot_condition_number_with_bars(scales_naive, cn_naive, cn_std_naive, scales_ortho, cn_ortho, cn_std_ortho, output_dir)
    plot_interference_with_bars(scales_naive, ii_naive, ii_std_naive, scales_ortho, ii_ortho, ii_std_ortho, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ All plots with error bars generated in {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

