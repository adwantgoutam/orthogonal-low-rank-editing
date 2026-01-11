"""
Download CounterFact and zsRE datasets for knowledge editing experiments.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List

# Dataset URLs (using HuggingFace datasets or direct links)
COUNTERFACT_URL = "https://raw.githubusercontent.com/rome-mem/counterfact/main/data/counterfact.json"
ZSRE_URL = "https://raw.githubusercontent.com/rome-mem/zsre/main/data/zsre.json"

def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False

def create_sample_counterfact(output_path: Path, num_samples: int = 100):
    """Create a sample CounterFact dataset for testing."""
    print(f"Creating sample CounterFact dataset with {num_samples} examples...")
    
    # Sample CounterFact format
    samples = []
    facts = [
        ("Paris", "capital of", "France", "Germany"),
        ("London", "capital of", "United Kingdom", "France"),
        ("Berlin", "capital of", "Germany", "Italy"),
        ("Madrid", "capital of", "Spain", "Portugal"),
        ("Rome", "capital of", "Italy", "Greece"),
        ("Tokyo", "capital of", "Japan", "China"),
        ("Beijing", "capital of", "China", "India"),
        ("Moscow", "capital of", "Russia", "Ukraine"),
        ("Washington", "capital of", "United States", "Canada"),
        ("Ottawa", "capital of", "Canada", "Mexico"),
    ]
    
    for i in range(num_samples):
        subject, relation, old_obj, new_obj = facts[i % len(facts)]
        
        sample = {
            "requested_rewrite": {
                "subject": subject,
                "relation_id": relation,
                "target_true": {"str": old_obj},
                "target_new": {"str": new_obj}
            },
            "neighborhood": [
                {"subject": "Berlin", "relation": "capital of", "object": "Germany"},
                {"subject": "Madrid", "relation": "capital of", "object": "Spain"}
            ],
            "paraphrase_prompts": [
                f"What is the capital of {subject}?",
                f"{subject} is the capital of which country?",
                f"Which country has {subject} as its capital?"
            ]
        }
        samples.append(sample)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Created sample dataset with {len(samples)} examples at {output_path}")

def create_sample_zsre(output_path: Path, num_samples: int = 100):
    """Create a sample zsRE dataset for testing."""
    print(f"Creating sample zsRE dataset with {num_samples} examples...")
    
    samples = []
    facts = [
        ("Paris", "capital", "France", "Germany"),
        ("London", "capital", "United Kingdom", "France"),
        ("Berlin", "capital", "Germany", "Italy"),
    ]
    
    for i in range(num_samples):
        subject, relation, old_obj, new_obj = facts[i % len(facts)]
        
        sample = {
            "subject": subject,
            "relation": relation,
            "old_answer": old_obj,
            "new_answer": new_obj,
            "other_answers": [
                {"subject": "Berlin", "relation": "capital", "object": "Germany"},
                {"subject": "Madrid", "relation": "capital", "object": "Spain"}
            ],
            "paraphrase_questions": [
                f"What is the capital of {subject}?",
                f"{subject} is the capital of which country?"
            ]
        }
        samples.append(sample)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Created sample dataset with {len(samples)} examples at {output_path}")

def main():
    """Download or create datasets."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    counterfact_path = data_dir / "counterfact.json"
    zsre_path = data_dir / "zsre.json"
    
    # Try to download CounterFact
    if not counterfact_path.exists():
        if not download_file(COUNTERFACT_URL, counterfact_path):
            print("Creating sample CounterFact dataset instead...")
            create_sample_counterfact(counterfact_path, num_samples=200)
    else:
        print(f"✓ CounterFact dataset already exists at {counterfact_path}")
    
    # Try to download zsRE
    if not zsre_path.exists():
        if not download_file(ZSRE_URL, zsre_path):
            print("Creating sample zsRE dataset instead...")
            create_sample_zsre(zsre_path, num_samples=200)
    else:
        print(f"✓ zsRE dataset already exists at {zsre_path}")
    
    print("\n✓ Dataset setup complete!")

if __name__ == "__main__":
    main()

