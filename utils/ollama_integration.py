"""
Ollama integration for paraphrase generation and evaluation.

This module provides utilities to use Ollama models for:
- Generating paraphrase variants
- Optional judge scoring
- Qualitative examples
"""

import requests
import json
from typing import List, Optional, Dict, Any


class OllamaParaphraser:
    """Use Ollama to generate paraphrases of queries."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama paraphraser.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use for paraphrasing
        """
        self.base_url = base_url
        self.model = model
    
    def generate_paraphrases(
        self,
        query: str,
        num_paraphrases: int = 5,
        subject: Optional[str] = None
    ) -> List[str]:
        """
        Generate paraphrases of a query.
        
        Args:
            query: Original query
            num_paraphrases: Number of paraphrases to generate
            subject: Optional subject to include in prompt
            
        Returns:
            List of paraphrased queries
        """
        prompt = f"""Generate {num_paraphrases} different ways to ask the following question. 
Return only the paraphrased questions, one per line, without numbering or bullets.

Original question: {query}
"""
        
        if subject:
            prompt += f"\nMake sure to include '{subject}' in the paraphrases."
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                paraphrases = result.get("response", "").strip().split("\n")
                # Clean up paraphrases
                paraphrases = [p.strip() for p in paraphrases if p.strip()]
                return paraphrases[:num_paraphrases]
            else:
                print(f"Warning: Ollama API returned status {response.status_code}")
                return self._fallback_paraphrases(query, num_paraphrases)
        except Exception as e:
            print(f"Warning: Ollama request failed: {e}")
            return self._fallback_paraphrases(query, num_paraphrases)
    
    def _fallback_paraphrases(self, query: str, num: int) -> List[str]:
        """Fallback paraphrases if Ollama is unavailable."""
        # Simple template-based paraphrases
        templates = [
            "What is {query}?",
            "Tell me about {query}",
            "Can you explain {query}?",
            "I want to know {query}",
            "Information about {query}"
        ]
        return [t.format(query=query) for t in templates[:num]]


class OllamaJudge:
    """Use Ollama as a judge for evaluating edit quality."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama judge.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use for judging
        """
        self.base_url = base_url
        self.model = model
    
    def score_edit_quality(
        self,
        subject: str,
        relation: str,
        expected_answer: str,
        model_answer: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score the quality of an edit using Ollama as a judge.
        
        Args:
            subject: Subject of the fact
            relation: Relation
            expected_answer: Expected answer after edit
            model_answer: Answer from the edited model
            context: Optional context
            
        Returns:
            Dictionary with score and reasoning
        """
        prompt = f"""Evaluate whether the model's answer is correct for the following question.

Question: {subject} {relation}
Expected answer: {expected_answer}
Model's answer: {model_answer}
"""
        
        if context:
            prompt += f"\nContext: {context}"
        
        prompt += "\n\nRate the answer on a scale of 0-1, where 1 is completely correct and 0 is completely wrong. Provide a brief explanation."
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Try to extract score
                score = self._extract_score(response_text)
                return {
                    "score": score,
                    "reasoning": response_text,
                    "model_answer": model_answer,
                    "expected_answer": expected_answer
                }
            else:
                return {"score": 0.5, "reasoning": "Ollama API unavailable", "error": True}
        except Exception as e:
            print(f"Warning: Ollama judge request failed: {e}")
            return {"score": 0.5, "reasoning": f"Error: {e}", "error": True}
    
    def _extract_score(self, text: str) -> float:
        """Extract numeric score from judge response."""
        import re
        # Look for numbers between 0 and 1
        matches = re.findall(r'\b0?\.\d+\b|\b1\.0\b', text)
        if matches:
            try:
                return float(matches[0])
            except:
                pass
        # Default to 0.5 if no score found
        return 0.5


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is available."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except:
        return []

