# Orthogonal Low-Rank Knowledge Editing

**Author**: gadwant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation for "Subspace Collisions in Knowledge Editing: Orthogonal Low-Rank Updates for Scalable, Stable Model Edits".

## Overview

This repository contains the implementation for the paper **"Subspace Collisions in Knowledge Editing: Orthogonal Low-Rank Updates for Scalable, Stable Model Edits"**

This research addresses the instability of existing low-rank knowledge editing methods (like ROME and MEMIT) when scaled to hundreds or thousands of edits. We identify "subspace collisions"—overlapping update directions in the model's representation space—as a primary cause of this instability.

The code provided here implements **Orthogonal Low-Rank Editing**, a novel approach that:
1.  **Enforces Orthogonality**: Ensures new knowledge updates are geometrically separated from existing ones.
2.  **Preserves Stability**: Maintains low condition numbers and high effective rank even as the number of edits scales.
3.  **Scales Effectively**: Demonstrates robust performance from 1 to 50+ edits where naive methods fail.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
code/
├── utils/
│   ├── orthogonal_editing.py    # Core orthogonal editing implementation
│   └── evaluation.py            # Evaluation metrics
├── scripts/
│   └── run_experiments.py       # Main experiment script
├── data/                        # Dataset storage
├── models/                      # Model checkpoints
├── notebooks/                   # Jupyter notebooks for analysis
└── requirements.txt             # Python dependencies
```

## Usage

### Running Experiments

```bash
python scripts/run_experiments.py \
    --model_name "EleutherAI/pythia-70m" \
    --dataset counterfact \
    --dataset_path data/counterfact.json \
    --output_dir results \
    --scales 1 3 5 10 25 50 \
    --use_orthogonal \
    --device cpu
```

### Key Components

#### OrthogonalLowRankEditor

The main class for applying orthogonal edits:

```python
from utils.orthogonal_editing import OrthogonalLowRankEditor, Edit

editor = OrthogonalLowRankEditor(model, tokenizer, use_qr=True, device="cpu")

# Apply a single edit
edit = Edit(
    subject="Paris",
    relation="capital of",
    old_object="France",
    new_object="Germany",
    layer_idx=6
)
# Returns u (update direction) and v (projection)
u, v = editor.apply_edit(edit)

# Apply multiple edits (automatically handles orthogonalization)
edits = [edit1, edit2, ...]
updates = editor.apply_edits_batch(edits)
# Apply to model weights
editor.apply_updates_to_model(updates)
```

#### Evaluation

```python
from utils.evaluation import KnowledgeEditingEvaluator

evaluator = KnowledgeEditingEvaluator(model, tokenizer, device="cpu")

result = evaluator.evaluate_edit(
    subject="Paris",
    relation="capital of",
    old_object="France",
    new_object="Germany",
    unrelated_facts=[...],
    paraphrases=[...]
)
```

## Datasets

### CounterFact

Download from: [CounterFact Dataset](https://github.com/rome-mem/counterfact)

### zsRE

Download from: [zsRE Dataset](https://github.com/rome-mem/zsre)

## Experiments

The paper experiments include:

1. **Scaling Analysis**: Testing edit performance from 1 to 50 edits.
2. **Baseline Comparison**: Comparing against ROME, MEMIT, and naive sequential editing.
3. **Geometric Analysis**: Measuring condition number, interference index, and effective rank.
4. **Robustness Testing**: Testing order invariance and noise robustness.

## Notes

- This implementation focuses on the geometric analysis of edit interactions.
- Designed for use with Pythia and GPT-style models.
- Uses `SimpleROME` (gradient-based rank-1 updates) as the base editor signal.

## Author

**gadwant**
- Initial implementation and experiments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

