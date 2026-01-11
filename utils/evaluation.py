"""
Evaluation metrics for knowledge editing.

This module implements standard evaluation metrics for knowledge editing:
- Edit Success (ES)
- Locality (L)
- Paraphrase Generalization (PG)
- Interference Index (II)
- Condition Number (CN)
- Effective Rank (ER)

@author: gadwant
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EditResult:
    """Results for a single edit."""
    edit_success: bool
    locality_score: float
    paraphrase_scores: List[float]
    subject: str
    relation: str
    old_object: str
    new_object: str


class KnowledgeEditingEvaluator:
    """
    Evaluator for knowledge editing methods.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: The language model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_edit_success(
        self,
        subject: str,
        relation: str,
        expected_object: str,
        prompt_template: str = "{subject} {relation}",
        max_new_tokens: int = 8
    ) -> bool:
        """
        Check if the model correctly predicts the expected object.
        
        Args:
            subject: Subject of the fact
            relation: Relation
            expected_object: Expected object (after edit)
            prompt_template: Template for generating prompts
            max_new_tokens: Maximum tokens to generate (conservative default: 8)
            
        Returns:
            True if edit was successful
        """
        prompt = prompt_template.format(subject=subject, relation=relation)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Conservative: disable KV cache
            )
        
        # Decode and check
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        
        # Check if expected object appears in generated text
        return expected_object.lower() in generated_text.lower()
    
    def compute_locality(
        self,
        unrelated_facts: List[Tuple[str, str, str]],
        prompt_template: str = "{subject} {relation}"
    ) -> float:
        """
        Compute locality score (preservation of unrelated facts).
        
        Args:
            unrelated_facts: List of (subject, relation, expected_object) tuples
            prompt_template: Template for generating prompts
            
        Returns:
            Locality score (percentage of facts preserved)
        """
        correct = 0
        total = len(unrelated_facts)
        
        for subject, relation, expected_object in unrelated_facts:
            if self.compute_edit_success(subject, relation, expected_object, prompt_template):
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_paraphrase_generalization(
        self,
        subject: str,
        relation: str,
        expected_object: str,
        paraphrases: List[str],
        max_new_tokens: int = 8
    ) -> List[float]:
        """
        Compute paraphrase generalization scores.
        
        Args:
            subject: Subject of the fact
            relation: Relation
            expected_object: Expected object
            paraphrases: List of paraphrased queries
            max_new_tokens: Maximum tokens to generate (conservative default: 8)
            
        Returns:
            List of scores (1.0 if correct, 0.0 otherwise) for each paraphrase
        """
        scores = []
        
        for paraphrase in paraphrases:
            # Replace subject and relation in paraphrase if needed
            prompt = paraphrase.format(subject=subject)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False  # Conservative: disable KV cache
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            score = 1.0 if expected_object.lower() in generated_text.lower() else 0.0
            scores.append(score)
        
        return scores
    
    def evaluate_edit(
        self,
        subject: str,
        relation: str,
        old_object: str,
        new_object: str,
        unrelated_facts: Optional[List[Tuple[str, str, str]]] = None,
        paraphrases: Optional[List[str]] = None,
        prompt_template: str = "{subject} {relation}",
        max_new_tokens: int = 8
    ) -> EditResult:
        """
        Evaluate a single edit comprehensively.
        
        Args:
            subject: Subject of the fact
            relation: Relation
            old_object: Original object (before edit)
            new_object: New object (after edit)
            unrelated_facts: List of unrelated facts to test locality
            paraphrases: List of paraphrased queries
            prompt_template: Template for generating prompts
            
        Returns:
            EditResult with all metrics
        """
        # Edit success
        edit_success = self.compute_edit_success(
            subject, relation, new_object, prompt_template, max_new_tokens
        )
        
        # Locality
        locality_score = 1.0
        if unrelated_facts:
            locality_score = self.compute_locality(unrelated_facts, prompt_template)
        
        # Paraphrase generalization
        paraphrase_scores = []
        if paraphrases:
            paraphrase_scores = self.compute_paraphrase_generalization(
                subject, relation, new_object, paraphrases, max_new_tokens
            )
        
        return EditResult(
            edit_success=edit_success,
            locality_score=locality_score,
            paraphrase_scores=paraphrase_scores,
            subject=subject,
            relation=relation,
            old_object=old_object,
            new_object=new_object
        )
    
    def evaluate_batch(
        self,
        edits: List[Dict],
        unrelated_facts_dict: Optional[Dict[int, List[Tuple[str, str, str]]]] = None,
        paraphrases_dict: Optional[Dict[int, List[str]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of edits and compute aggregate metrics.
        
        Args:
            edits: List of edit dictionaries with keys: subject, relation, old_object, new_object
            unrelated_facts_dict: Optional dict mapping edit index to unrelated facts
            paraphrases_dict: Optional dict mapping edit index to paraphrases
            
        Returns:
            Dictionary with aggregate metrics
        """
        results = []
        
        for i, edit in enumerate(edits):
            unrelated = unrelated_facts_dict.get(i) if unrelated_facts_dict else None
            paraphrases = paraphrases_dict.get(i) if paraphrases_dict else None
            
            result = self.evaluate_edit(
                subject=edit["subject"],
                relation=edit["relation"],
                old_object=edit["old_object"],
                new_object=edit["new_object"],
                unrelated_facts=unrelated,
                paraphrases=paraphrases
            )
            results.append(result)
        
        # Aggregate metrics
        edit_success_rate = sum(r.edit_success for r in results) / len(results)
        avg_locality = sum(r.locality_score for r in results) / len(results)
        
        all_paraphrase_scores = []
        for r in results:
            all_paraphrase_scores.extend(r.paraphrase_scores)
        avg_paraphrase = sum(all_paraphrase_scores) / len(all_paraphrase_scores) if all_paraphrase_scores else 0.0
        
        return {
            "edit_success": edit_success_rate,
            "locality": avg_locality,
            "paraphrase_generalization": avg_paraphrase,
            "num_edits": len(results)
        }


def compute_condition_number(update_directions: List[torch.Tensor]) -> float:
    """
    Compute the condition number of the update subspace.
    
    Args:
        update_directions: List of update direction vectors
        
    Returns:
        Condition number
    """
    if len(update_directions) < 2:
        return 1.0
    
    U = torch.stack(update_directions, dim=1)
    UUT = U.T @ U
    
    eigenvals = torch.linalg.eigvalsh(UUT)
    eigenvals = eigenvals[eigenvals > 1e-10]
    
    if len(eigenvals) == 0:
        return float('inf')
    
    return (eigenvals.max() / eigenvals.min()).item()


def compute_interference_index(update_directions: List[torch.Tensor]) -> float:
    """
    Compute the interference index (average cosine similarity).
    
    Args:
        update_directions: List of update direction vectors
        
    Returns:
        Average absolute cosine similarity
    """
    if len(update_directions) < 2:
        return 0.0
    
    total_similarity = 0.0
    count = 0
    
    for i in range(len(update_directions)):
        for j in range(i + 1, len(update_directions)):
            u_i = update_directions[i]
            u_j = update_directions[j]
            similarity = torch.abs(torch.dot(u_i, u_j))
            total_similarity += similarity.item()
            count += 1
    
    return total_similarity / count if count > 0 else 0.0


def compute_effective_rank(
    update_directions: List[torch.Tensor],
    threshold: float = 1e-6
) -> int:
    """
    Compute the effective rank of the update subspace.
    
    Args:
        update_directions: List of update direction vectors
        threshold: Threshold for singular values
        
    Returns:
        Effective rank
    """
    if not update_directions:
        return 0
    
    U = torch.stack(update_directions, dim=1)
    singular_values = torch.linalg.svdvals(U)
    
    return (singular_values > threshold).sum().item()

