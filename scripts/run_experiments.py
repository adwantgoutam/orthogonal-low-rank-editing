"""
Main script for running knowledge editing experiments.

This script implements the experimental protocol described in the paper:
- Scaling from 1 to 1000 edits
- Comparison with baselines (ROME, MEMIT, naive sequential)
- Evaluation on CounterFact and zsRE datasets

@author: gadwant
"""

import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.orthogonal_editing import OrthogonalLowRankEditor, Edit
from utils.evaluation import KnowledgeEditingEvaluator


def load_dataset(dataset_path: str, dataset_name: str) -> List[Dict]:
    """
    Load dataset (CounterFact or zsRE).
    
    Args:
        dataset_path: Path to dataset file
        dataset_name: Name of dataset ('counterfact' or 'zsre')
        
    Returns:
        List of edit examples
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    edits = []
    if dataset_name == "counterfact":
        for item in data:
            edits.append({
                "subject": item["requested_rewrite"]["subject"],
                "relation": item["requested_rewrite"]["relation_id"],
                "old_object": item["requested_rewrite"]["target_true"]["str"],
                "new_object": item["requested_rewrite"]["target_new"]["str"],
                "unrelated_facts": item.get("neighborhood", []),
                "paraphrases": item.get("paraphrase_prompts", [])
            })
    elif dataset_name == "zsre":
        for item in data:
            edits.append({
                "subject": item["subject"],
                "relation": item["relation"],
                "old_object": item["old_answer"],
                "new_object": item["new_answer"],
                "unrelated_facts": item.get("other_answers", []),
                "paraphrases": item.get("paraphrase_questions", [])
            })
    
    return edits


def run_scaling_experiment(
    model,
    tokenizer,
    edits: List[Dict],
    scales: List[int] = [1, 10, 100, 1000],
    use_orthogonal: bool = True,
    device: str = "cuda"
):
    """
    Run scaling experiment with different numbers of edits.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        edits: List of edit dictionaries
        scales: List of edit counts to test
        use_orthogonal: Whether to use orthogonal editing
        device: Device to run on
        
    Returns:
        Dictionary of results for each scale
    """
    results = {}
    
    for scale in scales:
        if scale > len(edits):
            print(f"Skipping scale {scale} (only {len(edits)} edits available)")
            continue
        
        print(f"\nRunning experiment with {scale} edits...")
        
        # Sample edits
        sampled_edits = edits[:scale]
        
        # Initialize editor
        editor = OrthogonalLowRankEditor(
            model=model,
            use_qr=True,
            device=device,
            tokenizer=tokenizer,
            use_orthogonal=use_orthogonal
        )
        
        # Convert to Edit objects (simplified - adjust layer_idx based on your model)
        edit_objects = [
            Edit(
                subject=e["subject"],
                relation=e["relation"],
                old_object=e["old_object"],
                new_object=e["new_object"],
                layer_idx=model.config.num_hidden_layers // 2  # Edit middle layer
            )
            for e in sampled_edits
        ]
        
        # Apply edits
        updates = editor.apply_edits_batch(edit_objects)
        editor.apply_updates_to_model(updates)
        
        # Evaluate
        evaluator = KnowledgeEditingEvaluator(model, tokenizer, device)
        
        unrelated_facts_dict = {}
        paraphrases_dict = {}
        for i, edit in enumerate(sampled_edits):
            if "unrelated_facts" in edit:
                unrelated_facts_dict[i] = edit["unrelated_facts"]
            if "paraphrases" in edit:
                paraphrases_dict[i] = edit["paraphrases"]
        
        eval_results = evaluator.evaluate_batch(
            sampled_edits,
            unrelated_facts_dict=unrelated_facts_dict if unrelated_facts_dict else None,
            paraphrases_dict=paraphrases_dict if paraphrases_dict else None
        )
        
        # Compute geometric metrics
        condition_number = editor.compute_condition_number()
        interference_index = editor.compute_interference_index()
        effective_rank = editor.compute_effective_rank()
        
        results[scale] = {
            **eval_results,
            "condition_number": condition_number,
            "interference_index": interference_index,
            "effective_rank": effective_rank
        }
        
        print(f"Scale {scale} results:")
        print(f"  Edit Success: {eval_results['edit_success']:.3f}")
        print(f"  Locality: {eval_results['locality']:.3f}")
        print(f"  Paraphrase Gen: {eval_results['paraphrase_generalization']:.3f}")
        print(f"  Condition Number: {condition_number:.3f}")
        print(f"  Interference Index: {interference_index:.3f}")
        print(f"  Effective Rank: {effective_rank}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run knowledge editing experiments")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b",
                       help="Model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["counterfact", "zsre"],
                       help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10, 100, 1000],
                       help="Scales (number of edits) to test")
    parser.add_argument("--use_orthogonal", action="store_true",
                       help="Use orthogonal editing (default: True)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
    )
    model.to(args.device)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    edits = load_dataset(args.dataset_path, args.dataset)
    print(f"Loaded {len(edits)} edits")
    
    # Run experiments
    print("\nRunning scaling experiments...")
    results = run_scaling_experiment(
        model=model,
        tokenizer=tokenizer,
        edits=edits,
        scales=args.scales,
        use_orthogonal=args.use_orthogonal,
        device=args.device
    )
    
    # Save results
    output_path = Path(args.output_dir) / f"{args.dataset}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

