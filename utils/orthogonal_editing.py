"""
Orthogonal Low-Rank Knowledge Editing

This module implements orthogonal low-rank editing for knowledge editing in language models.
The method enforces orthogonality between update directions to prevent subspace collisions.

@author: gadwant
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from .rome_impl import compute_u_v_gradient
from transformers import PreTrainedTokenizer


@dataclass
class Edit:
    """Represents a single knowledge edit."""
    subject: str
    relation: str
    old_object: str
    new_object: str
    layer_idx: int


class OrthogonalLowRankEditor:
    """
    Implements orthogonal low-rank editing for knowledge editing.
    
    The method maintains an orthogonal basis for update directions to ensure
    numerical stability and prevent interference between edits.
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_qr: bool = True,
        device: str = "cuda",
        use_orthogonal: bool = True,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize the orthogonal editor.
        
        Args:
            model: The language model to edit
            use_qr: Whether to use QR decomposition (more stable) or Gram-Schmidt
            device: Device to run computations on
            use_orthogonal: If False, skip orthogonalization (naive baseline mode)
        """
        self.model = model
        self.use_qr = use_qr
        self.device = device
        self.use_orthogonal = use_orthogonal
        self.tokenizer = tokenizer
        self.update_directions = []  # List of orthogonal update directions
        self.update_vectors = []     # Corresponding v vectors
        self.edits = []              # List of applied edits
        
    def compute_base_update(
        self,
        edit: Edit,
        layer_name: str = "mlp",
        use_rome: bool = True,
        rome_function: Optional[callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the base update direction using ROME/MEMIT-style method.
        
        This can use either the integrated ROME/MEMIT implementation or
        a simplified placeholder. For full experiments, integrate with
        actual ROME/MEMIT implementations.
        
        Args:
            edit: The edit to apply
            layer_name: Name of the layer to edit
            use_rome: Whether to use ROME-style computation
            rome_function: Optional ROME function from integration
            
        Returns:
            Tuple of (u, v) update vectors
        """
        # If ROME/MEMIT function is provided, use it
        if rome_function is not None:
            try:
                u_init, v_init = rome_function(
                    model=self.model,
                    subject=edit.subject,
                    relation=edit.relation,
                    target_new=edit.new_object,
                    target_true=edit.old_object,
                    layer_idx=edit.layer_idx,
                    device=self.device
                )
                return u_init, v_init
            except Exception as e:
                print(f"Warning: ROME/MEMIT integration unavailable or failed: {e}")
                print("Falling back to internal simple ROME implementation")
        
        # Use SimpleROME implementation if tokenizer is available
        if self.tokenizer is not None:
            try:
                u_grad, v_act = compute_u_v_gradient(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    subject=edit.subject,
                    relation=edit.relation,
                    target_new=edit.new_object,
                    layer_idx=edit.layer_idx,
                    layer_name=layer_name,
                    device=self.device
                )
                
                # Normalize u so scale is controlled (we want u to be a direction)
                if torch.norm(u_grad) > 1e-8:
                    u_grad = u_grad / torch.norm(u_grad)
                
                # Normalize v? No, ROME keeps v scale related to activation magnitude
                # But for our "direction basis", we typically normalize input vectors
                # Let's normalize both directions to keep subspace analysis clean
                # The magnitude would be absorbed into singular values anyway
                v_act = v_act / torch.norm(v_act)
                
                return u_grad, v_act
            except Exception as e:
                print(f"SimpleROME failed: {e}. Using deterministic fallback.")
        
        # Get the target layer
        layer = self._get_layer(edit.layer_idx, layer_name)
        
        # Handle GPT-2 MLP structure (has c_fc and c_proj Conv1D layers)
        if hasattr(layer, 'c_proj'):
            # GPT-2 MLP: use c_proj (output projection)
            # Conv1D weight shape is (out_channels, in_channels)
            target_layer = layer.c_proj
            # Ensure contiguous and float32 for alignment safety
            weight = target_layer.weight.data.contiguous().float()
            m, n = weight.shape  # (out_channels, in_channels)
        elif hasattr(layer, 'dense_h_to_4h') and hasattr(layer, 'dense_4h_to_h'):
            # GPT-NeoX / Pythia MLP structure: has dense_h_to_4h and dense_4h_to_h
            # Use dense_4h_to_h (output projection)
            target_layer = layer.dense_4h_to_h
            # Ensure contiguous and float32 for alignment safety
            weight = target_layer.weight.data.contiguous().float()
            m, n = weight.shape
        elif hasattr(layer, 'weight'):
            # Standard linear layer
            # Ensure contiguous and float32 for alignment safety
            weight = layer.weight.data.contiguous().float()
            m, n = weight.shape
        else:
            raise ValueError(f"Layer {type(layer)} structure not recognized. Expected GPT-2 MLP, GPT-NeoX MLP, or Linear layer.")
        
        # Simulated Update: Deterministic generation for geometric analysis
        # For full reproduction of language modeling results, integrate with ROME/MEMIT
        # This creates a deterministic update direction based on the edit content
        seed = hash(f"{edit.subject}_{edit.relation}_{edit.new_object}") % (2**32)
        torch.manual_seed(seed)
        u_init = torch.randn(m, device=self.device, dtype=torch.float32)
        v_init = torch.randn(n, device=self.device, dtype=torch.float32)
        
        # Ensure contiguous for alignment safety
        u_init = u_init.contiguous()
        v_init = v_init.contiguous()
        
        # Normalize
        u_init = u_init / torch.norm(u_init)
        v_init = v_init / torch.norm(v_init)
        
        return u_init, v_init
    
    def orthogonalize_gram_schmidt(
        self,
        u_new: torch.Tensor,
        existing_directions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Orthogonalize a new update direction against existing directions using Gram-Schmidt.
        
        Args:
            u_new: New update direction to orthogonalize
            existing_directions: List of existing orthogonal directions
            
        Returns:
            Orthogonalized and normalized update direction
        """
        # Ensure float32 and contiguous for alignment safety
        u_new = u_new.contiguous().float()
        u_ortho = u_new.clone()
        
        for u_existing in existing_directions:
            # Ensure existing direction is contiguous and float32
            u_existing = u_existing.contiguous().float()
            # Project out the component along existing direction
            projection = torch.dot(u_new, u_existing) / torch.dot(u_existing, u_existing)
            u_ortho = u_ortho - projection * u_existing
        
        # Normalize
        norm = torch.norm(u_ortho)
        if norm < 1e-8:
            raise ValueError("Update direction collapsed to zero after orthogonalization")
        
        u_ortho = u_ortho / norm
        return u_ortho.contiguous()
    
    def orthogonalize_qr(
        self,
        all_directions: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Orthogonalize all update directions using QR decomposition.
        
        Args:
            all_directions: List of all update directions (including new one)
            
        Returns:
            List of orthogonalized directions
        """
        if not all_directions:
            return []
        
        # Ensure all directions are contiguous and float32 for alignment safety
        all_directions = [d.contiguous().float() for d in all_directions]
        
        # Stack directions into a matrix
        U_matrix = torch.stack(all_directions, dim=1)  # Shape: (m, k)
        U_matrix = U_matrix.contiguous().float()
        
        # Compute QR decomposition
        Q, R = torch.linalg.qr(U_matrix)
        
        # Extract orthogonal columns (ensure contiguous)
        orthogonal_directions = [Q[:, i].contiguous() for i in range(Q.shape[1])]
        
        return orthogonal_directions
    
    def apply_edit(
        self,
        edit: Edit,
        layer_name: str = "mlp"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a single edit with optional orthogonalization.
        
        Args:
            edit: The edit to apply
            layer_name: Name of the layer to edit
            
        Returns:
            Tuple of (u, v) update vectors (orthogonalized if use_orthogonal=True)
        """
        # Compute base update direction
        u_init, v_init = self.compute_base_update(edit, layer_name)
        
        if not self.use_orthogonal:
            # Naive mode: use original direction without orthogonalization
            u_final = u_init
        elif self.use_qr and len(self.update_directions) > 0:
            # Use QR decomposition for all directions
            all_directions = self.update_directions + [u_init]
            orthogonal_directions = self.orthogonalize_qr(all_directions)
            u_final = orthogonal_directions[-1]
        else:
            # Use Gram-Schmidt for incremental orthogonalization
            u_final = self.orthogonalize_gram_schmidt(u_init, self.update_directions)
        
        # Store the direction (orthogonalized or not)
        self.update_directions.append(u_final)
        self.update_vectors.append(v_init)
        self.edits.append(edit)
        
        return u_final, v_init
    
    def apply_edits_batch(
        self,
        edits: List[Edit],
        layer_name: str = "mlp"
    ) -> Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Apply multiple edits in batch with orthogonalization.
        
        Returns (U, V) pairs instead of materializing full ΔW to save memory.
        
        Args:
            edits: List of edits to apply
            layer_name: Name of the layer to edit
            
        Returns:
            Dictionary mapping layer indices to (U_list, V_list) tuples
            where U_list and V_list are lists of vectors (not materialized matrices)
        """
        # Group edits by layer
        edits_by_layer: Dict[int, List[Edit]] = {}
        for edit in edits:
            if edit.layer_idx not in edits_by_layer:
                edits_by_layer[edit.layer_idx] = []
            edits_by_layer[edit.layer_idx].append(edit)
        
        updates = {}
        
        for layer_idx, layer_edits in edits_by_layer.items():
            # Reset for this layer
            self.update_directions = []
            self.update_vectors = []
            
            # Apply all edits for this layer
            for edit in layer_edits:
                self.apply_edit(edit, layer_name)
            
            # Return U and V lists (not materialized U @ V)
            if self.update_directions:
                # Ensure contiguous and float32 for alignment safety
                U_list = [d.contiguous().float() for d in self.update_directions]
                V_list = [v.contiguous().float() for v in self.update_vectors]
                updates[layer_idx] = (U_list, V_list)
        
        return updates
    
    def compute_condition_number(self) -> float:
        """
        Compute the condition number of the update subspace.
        
        Returns:
            Condition number (should be 1.0 for orthogonal directions)
        """
        if len(self.update_directions) < 2:
            return 1.0
        
        # Ensure contiguous and float32 for alignment safety
        directions = [d.contiguous().float() for d in self.update_directions]
        U = torch.stack(directions, dim=1)
        U = U.contiguous().float()
        UUT = (U.T @ U).contiguous()
        
        # Compute condition number
        eigenvals = torch.linalg.eigvalsh(UUT)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Filter near-zero eigenvalues
        
        if len(eigenvals) == 0:
            return float('inf')
        
        kappa = eigenvals.max() / eigenvals.min()
        return kappa.item()
    
    def compute_interference_index(self) -> float:
        """
        Compute the interference index between update directions.
        
        Returns:
            Average cosine similarity between all pairs of directions
        """
        if len(self.update_directions) < 2:
            return 0.0
        
        total_similarity = 0.0
        count = 0
        
        for i in range(len(self.update_directions)):
            for j in range(i + 1, len(self.update_directions)):
                # Ensure contiguous and float32 for alignment safety
                u_i = self.update_directions[i].contiguous().float()
                u_j = self.update_directions[j].contiguous().float()
                similarity = torch.abs(torch.dot(u_i, u_j))
                total_similarity += similarity.item()
                count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def compute_effective_rank(self, threshold: float = 1e-6) -> int:
        """
        Compute the effective rank of the update subspace.
        
        Args:
            threshold: Threshold for singular values
            
        Returns:
            Number of singular values above threshold
        """
        if not self.update_directions:
            return 0
        
        # Ensure contiguous and float32 for alignment safety
        directions = [d.contiguous().float() for d in self.update_directions]
        U = torch.stack(directions, dim=1)
        U = U.contiguous().float()
        singular_values = torch.linalg.svdvals(U)
        
        effective_rank = (singular_values > threshold).sum().item()
        return effective_rank
    
    def _get_layer(self, layer_idx: int, layer_name: str) -> nn.Module:
        """
        Get the target layer from the model.
        
        Args:
            layer_idx: Index of the layer
            layer_name: Name/type of the layer
            
        Returns:
            The target layer module
        """
        # GPT-2 / DistilGPT-2 architecture
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
                if layer_name == "mlp":
                    # GPT-2 MLP has c_fc and c_proj (Conv1D layers)
                    # We'll edit c_proj (output projection)
                    if hasattr(layer, 'mlp'):
                        mlp = layer.mlp
                        # Return the MLP module itself, we'll handle c_proj in compute_base_update
                        return mlp
                    else:
                        raise ValueError(f"Layer {layer_idx} does not have mlp attribute")
        elif hasattr(self.model, 'gpt_neox'):
            # GPT-NeoX / Pythia architecture
            layer = self.model.gpt_neox.layers[layer_idx]
            if layer_name == "mlp":
                if hasattr(layer, 'mlp'):
                    return layer.mlp
                else:
                    raise ValueError(f"Layer {layer_idx} does not have mlp attribute")
        
        raise ValueError(f"Could not find layer {layer_idx} with name {layer_name}")
    
    def apply_updates_to_model(self, updates: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]]):
        """
        Apply computed updates to the model weights using block outer-product updates.
        
        This avoids materializing the full ΔW matrix by applying each u_i @ v_i
        incrementally: W += u_1 @ v_1 + u_2 @ v_2 + ... = W + sum_i (u_i @ v_i)
        
        Args:
            updates: Dictionary mapping layer indices to (U_list, V_list) tuples
                    where U_list and V_list are lists of vectors
        """
        for layer_idx, (U_list, V_list) in updates.items():
            layer = self._get_layer(layer_idx, "mlp")
            with torch.no_grad():
                # Get weight matrix
                if hasattr(layer, 'c_proj'):
                    # GPT-2 MLP: apply to output projection (Conv1D)
                    weight = layer.c_proj.weight.data.contiguous().float()
                elif hasattr(layer, 'dense_4h_to_h'):
                    # GPT-NeoX / Pythia MLP: apply to output projection
                    weight = layer.dense_4h_to_h.weight.data.contiguous().float()
                elif hasattr(layer, 'weight'):
                    # Standard linear layer
                    weight = layer.weight.data.contiguous().float()
                else:
                    raise ValueError(f"Cannot apply update to layer {type(layer)}")
                
                # Apply updates incrementally: W += sum_i (u_i @ v_i)
                # This avoids materializing the full ΔW = U @ V matrix
                for u_i, v_i in zip(U_list, V_list):
                    # Ensure contiguous
                    u_i = u_i.contiguous().float()
                    v_i = v_i.contiguous().float()
                    # Block outer-product update: W += u_i @ v_i
                    # u_i is (m,), v_i is (n,), so u_i @ v_i is (m, n)
                    weight = weight + torch.outer(u_i, v_i)
                
                # Write back
                if hasattr(layer, 'c_proj'):
                    layer.c_proj.weight.data = weight.contiguous()
                elif hasattr(layer, 'dense_4h_to_h'):
                    layer.dense_4h_to_h.weight.data = weight.contiguous()
                elif hasattr(layer, 'weight'):
                    layer.weight.data = weight.contiguous()

