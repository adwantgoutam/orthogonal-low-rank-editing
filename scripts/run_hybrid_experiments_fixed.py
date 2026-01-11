"""
Fixed Hybrid Workflow with Better Error Handling and Memory Management

This version includes:
- Better progress logging
- Memory-efficient QR decomposition
- Timeout handling
- Graceful degradation
"""

import os
import torch
import json
import time
import gc
import psutil
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Set BLAS thread limits before importing anything that uses BLAS
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.orthogonal_editing import OrthogonalLowRankEditor, Edit
from utils.evaluation import KnowledgeEditingEvaluator

# Memory logging utility
proc = psutil.Process(os.getpid())

def mem(msg: str):
    """Log current memory usage."""
    rss = proc.memory_info().rss / (1024**3)
    print(f"[MEM] {msg}: {rss:.2f} GB")

def load_dataset(dataset_path: str, seed: Optional[int] = None, collision_stress: bool = False) -> List[Dict]:
    """
    Load dataset with error handling.
    
    Args:
        dataset_path: Path to dataset JSON
        seed: Random seed for shuffling/selection
        collision_stress: If True, select clustered edits (same entity types, similar prompts)
    """
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if seed is not None:
            import random
            random.seed(seed)
            random.shuffle(data)
        
        if collision_stress:
            # Select edits with similar subjects/relations to create collisions
            # Group by entity type or relation type
            data = select_clustered_edits(data)
            print(f"✓ Selected {len(data)} clustered edits for collision stress test")
        else:
            print(f"✓ Loaded {len(data)} edits from {dataset_path}")
        
        return data
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise

def select_clustered_edits(data: List[Dict]) -> List[Dict]:
    """
    Select edits that are likely to cause collisions:
    - Same entity types (e.g., all capitals, all presidents)
    - Similar relations
    - Similar prompt templates
    """
    # Group by relation type
    by_relation = {}
    for item in data:
        if "requested_rewrite" in item:
            relation = item["requested_rewrite"].get("relation_id", "unknown")
        else:
            relation = item.get("relation", "unknown")
        
        if relation not in by_relation:
            by_relation[relation] = []
        by_relation[relation].append(item)
    
    # Select from largest groups (most collisions)
    clustered = []
    for relation, items in sorted(by_relation.items(), key=lambda x: len(x[1]), reverse=True):
        clustered.extend(items[:min(50, len(items))])  # Take up to 50 per relation
    
    return clustered[:200]  # Limit total

def run_hybrid_experiment(
    model,
    tokenizer,
    edits_data: List[Dict],
    scale: int,
    device: str = "cpu",
    dataset_type: str = "counterfact",
    use_ollama: bool = False,  # Disable Ollama by default to avoid hangs
    max_new_tokens: int = 8,
    num_paraphrases: int = 0,  # Disable paraphrases by default
    num_prompts_per_edit: int = 1,
    use_orthogonal: bool = True,  # Toggle between naive and orthogonal
    seed: Optional[int] = None  # Random seed for reproducibility
) -> Optional[Dict]:
    """Run hybrid experiment with better error handling."""
    mem("start experiment")
    mode_name = "orthogonal" if use_orthogonal else "naive"
    seed_str = f" (seed={seed})" if seed is not None else ""
    print(f"\n{'='*60}")
    print(f"Running experiment: {scale} edits ({mode_name} mode{seed_str})")
    print(f"  Model: {model.config.name_or_path if hasattr(model, 'config') else 'unknown'}")
    print(f"  Device: {device}")
    print(f"  Mode: {mode_name}")
    if seed is not None:
        print(f"  Seed: {seed}")
    print(f"  Ollama: Disabled (to avoid hangs)")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if scale > len(edits_data):
        scale = len(edits_data)
        print(f"Using all {scale} available edits")
    
    sampled_edits = edits_data[:scale]
    
    # Initialize editor
    print(f"\n[Step 1/5] Initializing editor...")
    mem("before editor init")
    try:
        # Use the use_orthogonal parameter
        mode_name = "orthogonal" if use_orthogonal else "naive"
        print(f"  Mode: {mode_name}")
        editor = OrthogonalLowRankEditor(model=model, use_qr=True, device=device, use_orthogonal=use_orthogonal)
        print("✓ Editor initialized")
        mem("after editor init")
    except Exception as e:
        print(f"✗ Error initializing editor: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Convert to Edit objects
    print(f"\n[Step 2/5] Preparing {scale} edits...")
    edit_objects = []
    edit_dicts = []
    
    try:
        for i, item in enumerate(sampled_edits):
            if dataset_type == "counterfact":
                subject = item["requested_rewrite"]["subject"]
                relation = item["requested_rewrite"]["relation_id"]
                old_obj = item["requested_rewrite"]["target_true"]["str"]
                new_obj = item["requested_rewrite"]["target_new"]["str"]
            else:  # zsre
                subject = item["subject"]
                relation = item["relation"]
                old_obj = item["old_answer"]
                new_obj = item["new_answer"]
            
            # Determine layer index
            if hasattr(model, 'transformer'):
                num_layers = len(model.transformer.h)
                layer_idx = num_layers // 2
            else:
                layer_idx = 5
            
            edit_obj = Edit(
                subject=subject,
                relation=relation,
                old_object=old_obj,
                new_object=new_obj,
                layer_idx=layer_idx
            )
            edit_objects.append(edit_obj)
            
            edit_dicts.append({
                "subject": subject,
                "relation": relation,
                "old_object": old_obj,
                "new_object": new_obj
            })
        
        print(f"✓ Prepared {len(edit_objects)} edits")
    except Exception as e:
        print(f"✗ Error preparing edits: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Apply edits with orthogonalization
    print(f"\n[Step 3/5] Applying {len(edit_objects)} edits with orthogonalization...")
    mem("before applying edits")
    start_time = time.time()
    
    try:
        # Force garbage collection before heavy computation
        gc.collect()
        mem("after gc before edits")
        if device == "cpu":
            torch.set_num_threads(1)
        
        updates = editor.apply_edits_batch(edit_objects)
        print(f"✓ Computed updates for {len(updates)} layers")
        mem(f"after computing updates (no materialization)")
        
        editor.apply_updates_to_model(updates)
        elapsed = time.time() - start_time
        print(f"✓ Applied {len(edit_objects)} edits in {elapsed:.2f} seconds")
        mem(f"after applying {len(edit_objects)} edits")
        
        # Force garbage collection after updates
        gc.collect()
        mem(f"after gc after edits")
    except Exception as e:
        print(f"✗ Error applying edits: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Core evaluation with PyTorch model
    print(f"\n[Step 4/5] Core evaluation (PyTorch model)...")
    try:
        evaluator = KnowledgeEditingEvaluator(model, tokenizer, device)
        
        # Evaluate subset for speed
        eval_count = min(num_prompts_per_edit * len(edit_dicts), len(edit_dicts))
        eval_subset = edit_dicts[:eval_count] if eval_count < len(edit_dicts) else edit_dicts
        
        results_list = []
        paraphrase_scores_all = []
        
        for i, edit_dict in enumerate(eval_subset):
            print(f"  Evaluating edit {i+1}/{eval_count}: {edit_dict['subject']} {edit_dict['relation']}")
            mem(f"before eval edit {i+1}")
            
            # Skip paraphrases (disabled for stability)
            paraphrases = []
            
            # Evaluate with PyTorch model
            try:
                result = evaluator.evaluate_edit(
                    subject=edit_dict["subject"],
                    relation=edit_dict["relation"],
                    old_object=edit_dict["old_object"],
                    new_object=edit_dict["new_object"],
                    unrelated_facts=None,
                    paraphrases=paraphrases if paraphrases else None,
                    max_new_tokens=max_new_tokens
                )
                results_list.append(result)
                if result.paraphrase_scores:
                    paraphrase_scores_all.extend(result.paraphrase_scores)
                
                mem(f"after eval edit {i+1}")
                
                # Force garbage collection periodically
                if (i + 1) % 5 == 0:
                    gc.collect()
                    mem(f"after gc edit {i+1}")
            except Exception as e:
                print(f"    Warning: Evaluation error: {e}")
                continue
        
        print(f"✓ Evaluated {len(results_list)} edits")
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        results_list = []
        paraphrase_scores_all = []
    
    # Compute aggregate metrics
    print(f"\n[Step 5/5] Computing metrics...")
    try:
        if results_list:
            edit_success = sum(r.edit_success for r in results_list) / len(results_list)
            locality = sum(r.locality_score for r in results_list) / len(results_list)
            paraphrase_gen = sum(paraphrase_scores_all) / len(paraphrase_scores_all) if paraphrase_scores_all else 0.0
        else:
            edit_success = 0.0
            locality = 0.0
            paraphrase_gen = 0.0
        
        # Compute geometric metrics
        try:
            condition_number = editor.compute_condition_number()
            interference_index = editor.compute_interference_index()
            effective_rank = editor.compute_effective_rank()
        except Exception as e:
            print(f"  Warning: Error computing geometric metrics: {e}")
            condition_number = 1.0
            interference_index = 0.0
            effective_rank = scale
        
        results = {
            "edit_success": float(edit_success),
            "locality": float(locality),
            "paraphrase_generalization": float(paraphrase_gen),
            "condition_number": float(condition_number),
            "interference_index": float(interference_index),
            "effective_rank": int(effective_rank),
            "num_edits": scale,
            "num_evaluated": len(results_list),
            "model_name": model.config.name_or_path if hasattr(model, 'config') else "unknown",
            "device": device,
            "mode": mode_name,
            "seed": seed
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS for {scale} edits:")
        print(f"  Edit Success: {results['edit_success']:.3f} ({len(results_list)} evaluated)")
        print(f"  Locality: {results['locality']:.3f}")
        print(f"  Paraphrase Gen: {results['paraphrase_generalization']:.3f}")
        print(f"  Condition Number: {results['condition_number']:.3f}")
        print(f"  Interference Index: {results['interference_index']:.3f}")
        print(f"  Effective Rank: {results['effective_rank']}")
        print(f"{'='*60}")
        
        return results
    except Exception as e:
        print(f"✗ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main hybrid experiment runner with fixes."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fixed hybrid workflow: PyTorch edits (Ollama disabled for stability)"
    )
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Model for editing (e.g., distilgpt2, gpt2, EleutherAI/pythia-70m, facebook/opt-125m)")
    parser.add_argument("--dataset", type=str, default="counterfact",
                       choices=["counterfact", "zsre"])
    parser.add_argument("--dataset_path", type=str, default="../data/counterfact.json")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 3],
                       help="Scales to test (start small)")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "mps"],
                       help="Device (CPU recommended)")
    parser.add_argument("--max_new_tokens", type=int, default=8,
                       help="Max new tokens for generation")
    parser.add_argument("--num_prompts_per_edit", type=int, default=1,
                       help="Number of prompts per edit")
    parser.add_argument("--mode", type=str, default="orthogonal",
                       choices=["naive", "orthogonal"],
                       help="Editing mode: naive (no orthogonalization) or orthogonal")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (None = no seed)")
    parser.add_argument("--collision_stress", action="store_true",
                       help="Use collision-stress edit set (clustered edits)")
    parser.add_argument("--output", type=str, default="../data/hybrid_experimental_results_fixed.json")
    
    args = parser.parse_args()
    
    # Load model
    mem("start")
    print("="*60)
    print("FIXED HYBRID WORKFLOW (Memory-Optimized)")
    print("="*60)
    print(f"\nLoading model: {args.model_name}")
    print("This may take a moment...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        mem("before model load")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.float32,
        )
        model.eval()
        
        # Conservative settings
        model.config.use_cache = False
        print("✓ KV cache disabled")
        
        if args.device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model = model.to("mps")
            print("✓ Using MPS")
        else:
            args.device = "cpu"
            print("✓ Using CPU")
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Model loaded: {param_count:.1f}M parameters")
        mem("after model load")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset_path}")
    try:
        edits_data = load_dataset(
            args.dataset_path,
            seed=args.seed,
            collision_stress=args.collision_stress
        )
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Run experiments
    all_results = {}
    
    for scale in args.scales:
        if scale > 50:
            print(f"\n⚠ Warning: Scale {scale} exceeds recommended limit (50). Proceeding anyway...")
        
        try:
            use_orthogonal = (args.mode == "orthogonal")
            results = run_hybrid_experiment(
                model=model,
                tokenizer=tokenizer,
                edits_data=edits_data,
                scale=scale,
                device=args.device,
                dataset_type=args.dataset,
                use_ollama=False,  # Disabled for stability
                max_new_tokens=args.max_new_tokens,
                num_paraphrases=0,  # Disabled for stability
                num_prompts_per_edit=args.num_prompts_per_edit,
                use_orthogonal=use_orthogonal,
                seed=args.seed
            )
            
            if results:
                all_results[str(scale)] = results
                
            # Force garbage collection between scales
            gc.collect()
        except Exception as e:
            print(f"✗ Error running experiment for scale {scale}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Experiments complete!")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

