"""
Run experiments in batches with progress tracking.
"""

import subprocess
import sys
import os
from pathlib import Path

# Set environment
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def run_experiment(model, scales, mode, seed, output):
    """Run a single experiment."""
    cmd = [
        sys.executable, "run_hybrid_experiments_fixed.py",
        "--model_name", model,
        "--scales"] + [str(s) for s in scales] + [
        "--device", "cpu",
        "--mode", mode,
        "--seed", str(seed),
        "--output", output
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {model} | {mode} | seed={seed} | scales={scales}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    """Run all experiments."""
    experiments = [
        # Pythia-70m: Naive (3 seeds)
        ("EleutherAI/pythia-70m", [1, 3, 5, 10, 25, 50], "naive", 42, "../data/pythia70m_naive_seed42.json"),
        ("EleutherAI/pythia-70m", [1, 3, 5, 10, 25, 50], "naive", 123, "../data/pythia70m_naive_seed123.json"),
        ("EleutherAI/pythia-70m", [1, 3, 5, 10, 25, 50], "naive", 456, "../data/pythia70m_naive_seed456.json"),
        
        # Pythia-70m: Orthogonal (3 seeds)
        ("EleutherAI/pythia-70m", [1, 3, 5, 10, 25, 50], "orthogonal", 42, "../data/pythia70m_orthogonal_seed42.json"),
        ("EleutherAI/pythia-70m", [1, 3, 5, 10, 25, 50], "orthogonal", 123, "../data/pythia70m_orthogonal_seed123.json"),
        ("EleutherAI/pythia-70m", [1, 3, 5, 10, 25, 50], "orthogonal", 456, "../data/pythia70m_orthogonal_seed456.json"),
        
        # Pythia-160m: Naive (3 seeds)
        ("EleutherAI/pythia-160m", [1, 3, 5, 10, 25], "naive", 42, "../data/pythia160m_naive_seed42.json"),
        ("EleutherAI/pythia-160m", [1, 3, 5, 10, 25], "naive", 123, "../data/pythia160m_naive_seed123.json"),
        ("EleutherAI/pythia-160m", [1, 3, 5, 10, 25], "naive", 456, "../data/pythia160m_naive_seed456.json"),
        
        # Pythia-160m: Orthogonal (3 seeds)
        ("EleutherAI/pythia-160m", [1, 3, 5, 10, 25], "orthogonal", 42, "../data/pythia160m_orthogonal_seed42.json"),
        ("EleutherAI/pythia-160m", [1, 3, 5, 10, 25], "orthogonal", 123, "../data/pythia160m_orthogonal_seed123.json"),
        ("EleutherAI/pythia-160m", [1, 3, 5, 10, 25], "orthogonal", 456, "../data/pythia160m_orthogonal_seed456.json"),
    ]
    
    print(f"Total experiments: {len(experiments)}")
    
    successful = 0
    failed = 0
    
    for i, (model, scales, mode, seed, output) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        if run_experiment(model, scales, mode, seed, output):
            successful += 1
            print(f"✓ Success")
        else:
            failed += 1
            print(f"✗ Failed")
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

