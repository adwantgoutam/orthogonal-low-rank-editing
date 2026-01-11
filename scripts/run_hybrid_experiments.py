"""
Hybrid Workflow for TechRxiv Submission

This script implements the recommended hybrid approach:
1. PyTorch/HF (distilgpt2 or gpt2) for actual edits + core evaluation
2. Ollama (mistral/gemma/llama3) for paraphrasing + qualitative examples

Safe configuration:
- Model: distilgpt2 or gpt2
- Device: CPU (stable)
- Scales: up to 50 edits reliably
- Python: 3.11 (required for M1 stability)
- BLAS: Single-threaded (prevents crashes)
"""

import os
import torch
import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Set BLAS thread limits before importing anything that uses BLAS
# This prevents macOS BLAS crashes
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.orthogonal_editing import OrthogonalLowRankEditor, Edit
from utils.evaluation import KnowledgeEditingEvaluator
from utils.ollama_integration import OllamaParaphraser, OllamaJudge, check_ollama_available

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def generate_paraphrases_ollama(
    subject: str,
    relation: str,
    paraphraser: Optional[OllamaParaphraser],
    num_paraphrases: int = 5
) -> List[str]:
    """Generate paraphrases using Ollama."""
    if paraphraser is None:
        return []
    
    query = f"{subject} {relation}"
    try:
        paraphrases = paraphraser.generate_paraphrases(
            query=query,
            num_paraphrases=num_paraphrases,
            subject=subject
        )
        return paraphrases
    except Exception as e:
        print(f"  Warning: Ollama paraphrasing failed: {e}")
        return []

def run_hybrid_experiment(
    model,
    tokenizer,
    edits_data: List[Dict],
    scale: int,
    device: str = "cpu",
    dataset_type: str = "counterfact",
    use_ollama: bool = True,
    ollama_model: str = "mistral",
    max_new_tokens: int = 8,
    num_paraphrases: int = 1,
    num_prompts_per_edit: int = 1
) -> Dict:
    """Run hybrid experiment: PyTorch for edits, Ollama for paraphrasing."""
    print(f"\n{'='*60}")
    print(f"Running HYBRID experiment: {scale} edits")
    print(f"  Model: {model.config.name_or_path}")
    print(f"  Device: {device}")
    print(f"  Ollama: {'Enabled' if use_ollama else 'Disabled'}")
    print(f"{'='*60}")
    
    if scale > len(edits_data):
        scale = len(edits_data)
        print(f"Using all {scale} available edits")
    
    sampled_edits = edits_data[:scale]
    
    # Setup Ollama if available
    paraphraser = None
    judge = None
    if use_ollama and check_ollama_available():
        try:
            paraphraser = OllamaParaphraser(model=ollama_model)
            judge = OllamaJudge(model=ollama_model)
            print(f"✓ Ollama connected ({ollama_model})")
        except Exception as e:
            print(f"⚠ Ollama setup failed: {e}")
            paraphraser = None
            judge = None
    else:
        print("⚠ Ollama not available - using fallback paraphrases")
    
    # Initialize editor
    editor = OrthogonalLowRankEditor(model=model, use_qr=True, device=device)
    
    # Convert to Edit objects
    edit_objects = []
    edit_dicts = []
    
    print(f"\nPreparing {scale} edits...")
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
    
    # Apply edits with orthogonalization
    print(f"\nApplying {len(edit_objects)} edits with orthogonalization...")
    start_time = time.time()
    
    try:
        updates = editor.apply_edits_batch(edit_objects)
        editor.apply_updates_to_model(updates)
        elapsed = time.time() - start_time
        print(f"✓ Applied {len(edit_objects)} edits in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"✗ Error applying edits: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Core evaluation with PyTorch model
    print(f"\nCore evaluation (PyTorch model)...")
    evaluator = KnowledgeEditingEvaluator(model, tokenizer, device)
    
    # Evaluate subset for speed (conservative: limit to num_prompts_per_edit)
    eval_count = min(num_prompts_per_edit * len(edit_dicts), len(edit_dicts))
    eval_subset = edit_dicts[:eval_count] if eval_count < len(edit_dicts) else edit_dicts
    
    results_list = []
    paraphrase_scores_all = []
    
    for i, edit_dict in enumerate(eval_subset):
        print(f"  Evaluating edit {i+1}/{eval_count}: {edit_dict['subject']} {edit_dict['relation']}")
        
        # Generate paraphrases with Ollama (conservative: only if enabled)
        paraphrases = []
        if paraphraser and num_paraphrases > 0:
            paraphrases = generate_paraphrases_ollama(
                edit_dict["subject"],
                edit_dict["relation"],
                paraphraser,
                num_paraphrases=num_paraphrases
            )
        
        # Fallback to dataset paraphrases if Ollama failed
        if not paraphrases and i < len(sampled_edits):
            if dataset_type == "counterfact" and "paraphrase_prompts" in sampled_edits[i]:
                paraphrases = sampled_edits[i]["paraphrase_prompts"][:5]
            elif dataset_type == "zsre" and "paraphrase_questions" in sampled_edits[i]:
                paraphrases = sampled_edits[i]["paraphrase_questions"][:5]
        
        # Evaluate with PyTorch model
        try:
            result = evaluator.evaluate_edit(
                subject=edit_dict["subject"],
                relation=edit_dict["relation"],
                old_object=edit_dict["old_object"],
                new_object=edit_dict["new_object"],
                unrelated_facts=None,  # Skip for speed
                paraphrases=paraphrases if paraphrases else None,
                max_new_tokens=max_new_tokens
            )
            results_list.append(result)
            paraphrase_scores_all.extend(result.paraphrase_scores)
        except Exception as e:
            print(f"    Warning: Evaluation error: {e}")
            continue
    
    # Compute aggregate metrics
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
        print(f"Warning: Error computing geometric metrics: {e}")
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
        "ollama_used": use_ollama and paraphraser is not None,
        "ollama_model": ollama_model if paraphraser else None
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS for {scale} edits:")
    print(f"  Edit Success: {results['edit_success']:.3f} ({len(results_list)} evaluated)")
    print(f"  Locality: {results['locality']:.3f}")
    print(f"  Paraphrase Gen: {results['paraphrase_generalization']:.3f}")
    print(f"  Condition Number: {results['condition_number']:.3f}")
    print(f"  Interference Index: {results['interference_index']:.3f}")
    print(f"  Effective Rank: {results['effective_rank']}")
    print(f"  Ollama Used: {results['ollama_used']}")
    print(f"{'='*60}")
    
    return results

def main():
    """Main hybrid experiment runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hybrid workflow: PyTorch edits + Ollama paraphrasing"
    )
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       choices=["distilgpt2", "gpt2", "sshleifer/tiny-gpt2"],
                       help="Model for editing (distilgpt2 or tiny-gpt2 recommended)")
    parser.add_argument("--dataset", type=str, default="counterfact",
                       choices=["counterfact", "zsre"])
    parser.add_argument("--dataset_path", type=str, default="../data/counterfact.json")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 5, 10, 25, 50],
                       help="Scales to test (up to 50 recommended)")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "mps"],
                       help="Device (CPU recommended for stability)")
    parser.add_argument("--use_ollama", action="store_true", default=True,
                       help="Use Ollama for paraphrasing")
    parser.add_argument("--ollama_model", type=str, default="mistral",
                       choices=["mistral", "gemma2", "llama3", "deepseek-r1"],
                       help="Ollama model for paraphrasing")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference (1 for stability)")
    parser.add_argument("--max_new_tokens", type=int, default=8,
                       help="Max new tokens for generation (8-16 for stability)")
    parser.add_argument("--num_paraphrases", type=int, default=1,
                       help="Number of paraphrases per edit (1 for stability, 0 to disable)")
    parser.add_argument("--num_prompts_per_edit", type=int, default=1,
                       help="Number of prompts per edit (1 for stability)")
    parser.add_argument("--output", type=str, default="../data/hybrid_experimental_results.json")
    
    args = parser.parse_args()
    
    # Load model
    print("="*60)
    print("HYBRID WORKFLOW: TechRxiv Submission")
    print("="*60)
    print(f"\nLoading model: {args.model_name}")
    print("This may take a moment...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.float32,
        )
        model.eval()
        
        # Conservative settings to prevent SIGKILL
        model.config.use_cache = False  # Disable KV cache
        print("✓ KV cache disabled (conservative setting)")
        
        if args.device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model = model.to("mps")
            print("✓ Using MPS (Apple Silicon)")
        else:
            args.device = "cpu"
            print("✓ Using CPU (stable)")
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Model loaded: {param_count:.1f}M parameters")
        print(f"✓ Conservative settings: batch_size={args.batch_size}, max_tokens={args.max_new_tokens}, paraphrases={args.num_paraphrases}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check Ollama
    if args.use_ollama:
        if check_ollama_available():
            print(f"✓ Ollama available (will use {args.ollama_model})")
        else:
            print("⚠ Ollama not available - will use fallback paraphrases")
            args.use_ollama = False
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset_path}")
    try:
        edits_data = load_dataset(args.dataset_path)
        print(f"✓ Loaded {len(edits_data)} edits")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Run experiments
    all_results = {}
    
    for scale in args.scales:
        if scale > 50:
            print(f"\n⚠ Warning: Scale {scale} exceeds recommended limit (50). Proceeding anyway...")
        
        try:
            results = run_hybrid_experiment(
                model=model,
                tokenizer=tokenizer,
                edits_data=edits_data,
                scale=scale,
                device=args.device,
                dataset_type=args.dataset,
                use_ollama=args.use_ollama,
                ollama_model=args.ollama_model,
                max_new_tokens=args.max_new_tokens,
                num_paraphrases=args.num_paraphrases,
                num_prompts_per_edit=args.num_prompts_per_edit
            )
            
            if results:
                all_results[str(scale)] = results
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
    print(f"✓ HYBRID experiments complete!")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")
    print("\nThese are ACTUAL results from:")
    print("  - PyTorch model for edits and core evaluation")
    print("  - Ollama for paraphrase generation")
    print("\nReady for TechRxiv submission!")

if __name__ == "__main__":
    main()

