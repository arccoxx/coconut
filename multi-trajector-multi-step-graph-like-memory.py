#!/usr/bin/env python3
"""
Generalized Multi-Trajectory COCONUT - Final Clean Implementation
==================================================================
A general reasoning framework where trajectories naturally discover 
their own specializations through training, without pre-specified strategies.
Builds upon COCONUT from Facebook research.

Key Features:
- Multiple trajectories explore reasoning space in parallel
- Natural specialization emerges through diversity rewards
- Graph memory enables rich inter-trajectory relationships
- No domain-specific assumptions - pure general reasoning
- Comprehensive accuracy tracking and distribution analysis

Usage:
    python coconut.py              # Normal training
    python coconut.py --debug       # Debug mode with minimal samples
    python coconut.py --test        # Test single sample

Author: Aidan Collins, Claude, Grok
Date: 2025
"""

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import os
import gc
import random
import re
import json
import sys
import traceback
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from datetime import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import tqdm based on environment
if 'get_ipython' in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# ============================================================
# DATA CLASSES AND HELPERS
# ============================================================

@dataclass
class TrajectoryState:
    """Encapsulates trajectory state with emergent specialization tracking"""
    position: torch.Tensor
    value: float
    confidence: float
    trajectory_id: int
    step_count: int
    is_active: bool = True
    parent_id: Optional[int] = None
    
    # Emergent specialization attributes (discovered, not prescribed)
    discovered_strategy: Optional[str] = None  # Will be learned, not set
    exploration_temperature: float = 1.0  # Each trajectory has its own temperature
    risk_tolerance: float = 0.5  # Learned preference for exploration vs exploitation
    
    def ensure_batch_dim(self, batch_size: int = 1):
        """Ensure position has proper batch dimension"""
        if self.position.dim() == 1:
            self.position = self.position.unsqueeze(0)
        if self.position.size(0) != batch_size:
            self.position = self.position.expand(batch_size, -1)
        return self


class DimensionHelper:
    """Utility class for consistent dimension handling"""
    
    @staticmethod
    def ensure_batch_dim(tensor: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """Ensure tensor has proper batch dimension"""
        if tensor is None:
            return None
        
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        current_batch = tensor.size(0)
        if current_batch != batch_size:
            if current_batch == 1:
                tensor = tensor.expand(batch_size, *tensor.shape[1:])
            elif batch_size == 1:
                tensor = tensor[:1]
            else:
                raise ValueError(f"Batch size mismatch: tensor has {current_batch}, expected {batch_size}")
        
        return tensor
    
    @staticmethod
    def safe_cat(tensors: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
        """Safely concatenate tensors with dimension checking"""
        if not tensors:
            return None
        
        tensors = [t for t in tensors if t is not None]
        if not tensors:
            return None
        
        max_dim = max(t.dim() for t in tensors)
        aligned = []
        for t in tensors:
            while t.dim() < max_dim:
                t = t.unsqueeze(0)
            aligned.append(t)
        
        return torch.cat(aligned, dim=dim)
    
    @staticmethod
    def safe_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Safely stack tensors with dimension checking"""
        if not tensors:
            return None
        
        ref_shape = tensors[0].shape
        for i, t in enumerate(tensors):
            if t.shape != ref_shape:
                if t.dim() == ref_shape[0] - 1:
                    t = t.unsqueeze(0)
                tensors[i] = t
        
        return torch.stack(tensors, dim=dim)


# ============================================================
# TRAJECTORY SPECIALIZATION SYSTEM
# ============================================================

class TrajectorySpecializationTracker:
    """
    Tracks and encourages natural specialization of trajectories.
    No pre-specified roles - trajectories discover their own strategies.
    """
    
    def __init__(self, num_trajectories: int, feature_dim: int, device=None):
        self.num_trajectories = num_trajectories
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        
        # Track trajectory behavioral patterns (emergent, not prescribed)
        self.trajectory_signatures = defaultdict(lambda: {
            'exploration_pattern': deque(maxlen=100),
            'decision_pattern': deque(maxlen=100),
            'success_contexts': deque(maxlen=50),
            'failure_contexts': deque(maxlen=50),
            'avg_step_size': deque(maxlen=100),
            'direction_variance': deque(maxlen=100),
            'confidence_profile': deque(maxlen=100),
        })
        
        self.discovered_roles = []
        self.trajectory_role_affinity = defaultdict(lambda: defaultdict(float))
    
    def update_trajectory_signature(
        self,
        traj_idx: int,
        step_size: float,
        direction: torch.Tensor,
        value: float,
        confidence: float,
        context: torch.Tensor
    ):
        """Update trajectory's behavioral signature"""
        sig = self.trajectory_signatures[traj_idx]
        
        # Record behavioral patterns
        sig['avg_step_size'].append(step_size)
        sig['confidence_profile'].append(confidence)
        
        # Track exploration diversity - keep everything on same device
        if len(sig['exploration_pattern']) > 0:
            last_dir = sig['exploration_pattern'][-1]
            if isinstance(last_dir, torch.Tensor):
                # Move last_dir to same device as direction for comparison
                last_dir_device = last_dir.to(direction.device)
                dir_change = 1.0 - F.cosine_similarity(
                    direction.unsqueeze(0), 
                    last_dir_device.unsqueeze(0)
                ).item()
                sig['direction_variance'].append(dir_change)
        
        # Store on CPU to save GPU memory
        sig['exploration_pattern'].append(direction.detach().cpu())
        
        # Context-based success/failure tracking
        if value > 0.5:
            sig['success_contexts'].append(context.detach().cpu())
        else:
            sig['failure_contexts'].append(context.detach().cpu())
    
    def compute_specialization_bonus(self, traj_idx: int) -> float:
        """
        Compute bonus for trajectory based on how specialized/unique it is.
        Encourages trajectories to develop distinct strategies.
        """
        sig = self.trajectory_signatures[traj_idx]
        
        if len(sig['avg_step_size']) < 10:
            return 0.0
        
        # Compute trajectory's unique characteristics
        avg_step = np.mean(list(sig['avg_step_size']))
        step_consistency = 1.0 / (np.std(list(sig['avg_step_size'])) + 1e-6)
        avg_confidence = np.mean(list(sig['confidence_profile']))
        
        # Compare with other trajectories
        uniqueness = 0.0
        for other_idx in range(self.num_trajectories):
            if other_idx == traj_idx:
                continue
            
            other_sig = self.trajectory_signatures[other_idx]
            if len(other_sig['avg_step_size']) < 10:
                continue
            
            # Measure behavioral difference
            other_avg_step = np.mean(list(other_sig['avg_step_size']))
            other_confidence = np.mean(list(other_sig['confidence_profile']))
            
            diff = abs(avg_step - other_avg_step) + abs(avg_confidence - other_confidence)
            uniqueness += diff
        
        # Normalize and return as bonus
        return min(uniqueness / max(1, self.num_trajectories - 1), 1.0) * 0.2
    
    def get_diversity_matrix(self) -> torch.Tensor:
        """Get matrix of pairwise trajectory diversity"""
        diversity = torch.zeros(self.num_trajectories, self.num_trajectories, device=self.device)
        
        for i in range(self.num_trajectories):
            for j in range(i + 1, self.num_trajectories):
                sig_i = self.trajectory_signatures[i]
                sig_j = self.trajectory_signatures[j]
                
                if len(sig_i['avg_step_size']) > 0 and len(sig_j['avg_step_size']) > 0:
                    # Compare behavioral patterns
                    step_diff = abs(np.mean(list(sig_i['avg_step_size'])) - 
                                  np.mean(list(sig_j['avg_step_size'])))
                    conf_diff = abs(np.mean(list(sig_i['confidence_profile'])) - 
                                  np.mean(list(sig_j['confidence_profile'])))
                    
                    diversity[i, j] = diversity[j, i] = step_diff + conf_diff
        
        return diversity


# ============================================================
# GRAPH MEMORY BANK
# ============================================================

class GraphMemoryBank(nn.Module):
    """
    Graph-based memory for general reasoning patterns.
    No domain-specific assumptions - learns general reasoning strategies.
    """
    
    def __init__(
        self,
        memory_size: int,
        reasoning_dim: int,
        num_heads: int = 4,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.memory_size = memory_size
        self.reasoning_dim = reasoning_dim
        self.num_heads = num_heads
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.device = device
        
        # Graph structure
        self.register_buffer('nodes', torch.zeros(memory_size, reasoning_dim, dtype=self.dtype))
        self.register_buffer('node_values', torch.full((memory_size,), -float('inf'), dtype=self.dtype))
        self.register_buffer('node_timestamps', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('adjacency_matrix', torch.zeros(memory_size, memory_size, dtype=self.dtype))
        self.register_buffer('node_creators', torch.zeros(memory_size, dtype=torch.long))
        
        # Graph attention for relationship discovery
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=reasoning_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
            dtype=self.dtype
        )
        
        # Relationship encoder
        self.relation_encoder = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim // 2, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1, dtype=self.dtype),
            nn.Sigmoid()
        )
        
        # Memory evolution network
        self.memory_evolution = nn.GRUCell(reasoning_dim, reasoning_dim, dtype=self.dtype)
        
        # Pattern recognition network
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, reasoning_dim // 4, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 4, 32, dtype=self.dtype)
        )
        
        self.memory_ptr = 0
        self.total_writes = 0
        self.memory_filled = False
        
        if device:
            self.to(device)
    
    def write(
        self,
        position: torch.Tensor,
        value: torch.Tensor,
        source_trajectory: int,
        related_positions: Optional[List[torch.Tensor]] = None
    ) -> int:
        """Write to memory and establish graph relationships"""
        try:
            position = DimensionHelper.ensure_batch_dim(position, 1).squeeze(0)
            
            # Validate position dimension
            if position.size(-1) != self.reasoning_dim:
                print(f"Warning: Position dimension {position.size(-1)} doesn't match reasoning_dim {self.reasoning_dim}")
                return -1
            
            with torch.no_grad():
                # Write node
                self.nodes[self.memory_ptr] = position.detach().to(self.dtype)
                
                if value.dim() > 0:
                    value = value.view(-1)[0]
                self.node_values[self.memory_ptr] = value.detach().to(self.dtype)
                self.node_timestamps[self.memory_ptr] = self.total_writes
                self.node_creators[self.memory_ptr] = source_trajectory
                
                current_idx = self.memory_ptr
                
                # Establish relationships
                if related_positions and len(related_positions) > 0:
                    for related_pos in related_positions:
                        if related_pos is not None:
                            related_pos = DimensionHelper.ensure_batch_dim(related_pos, 1).squeeze(0)
                            
                            # Validate dimension
                            if related_pos.size(-1) != self.reasoning_dim:
                                continue
                            
                            similarities = F.cosine_similarity(
                                related_pos.unsqueeze(0),
                                self.nodes,
                                dim=1
                            )
                            
                            k = min(3, self.memory_ptr if not self.memory_filled else self.memory_size)
                            if k > 0:
                                top_k = torch.topk(similarities, k)
                                for idx in top_k.indices:
                                    if idx != current_idx:
                                        combined = torch.cat([
                                            self.nodes[current_idx],
                                            self.nodes[idx]
                                        ])
                                        strength = self.relation_encoder(combined).item()
                                        
                                        self.adjacency_matrix[current_idx, idx] = strength
                                        self.adjacency_matrix[idx, current_idx] = strength
                
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
                if self.memory_ptr == 0:
                    self.memory_filled = True
                
                self.total_writes += 1
                
                return current_idx
                
        except Exception as e:
            print(f"Warning: GraphMemoryBank write failed: {e}")
            return -1
    
    def retrieve_graph_context(
        self,
        query: torch.Tensor,
        num_hops: int = 2,
        top_k: int = 5,
        source_trajectory: Optional[int] = None,
        debug_mode: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Retrieve context using graph traversal"""
        try:
            query = DimensionHelper.ensure_batch_dim(query, 1)
            
            # Validate query dimension
            if query.size(-1) != self.reasoning_dim:
                if debug_mode:
                    print(f"Warning: Query dimension {query.size(-1)} doesn't match reasoning_dim {self.reasoning_dim}")
                return None, {}
            
            if not self.memory_filled and self.memory_ptr < top_k:
                return None, {}
            
            valid_nodes = self.memory_ptr if not self.memory_filled else self.memory_size
            
            # Initial retrieval
            query_norm = F.normalize(query, p=2, dim=-1)
            nodes_norm = F.normalize(self.nodes[:valid_nodes], p=2, dim=-1)
            similarities = torch.matmul(query_norm, nodes_norm.t()).squeeze(0)
            
            # Weight by value and recency
            recency_weight = torch.exp(-0.01 * (self.total_writes - self.node_timestamps[:valid_nodes]).float())
            value_weight = torch.sigmoid(self.node_values[:valid_nodes])
            
            # Diversity bonus
            diversity_bonus = torch.ones(valid_nodes, device=self.device, dtype=self.dtype)
            if source_trajectory is not None:
                different_creator = (self.node_creators[:valid_nodes] != source_trajectory).float()
                diversity_bonus += different_creator * 0.2
            
            weighted_sim = similarities * value_weight * recency_weight * diversity_bonus
            
            # Get initial nodes
            k = min(top_k, valid_nodes)
            top_k_scores, top_k_indices = torch.topk(weighted_sim, k)
            
            # Graph traversal
            visited = set(top_k_indices.tolist())
            context_nodes = list(visited)
            
            for hop in range(num_hops):
                new_nodes = set()
                for node_idx in list(visited):
                    connections = self.adjacency_matrix[node_idx, :valid_nodes]
                    connected = torch.where(connections > 0.3)[0]
                    
                    for conn_idx in connected:
                        if conn_idx.item() not in visited:
                            new_nodes.add(conn_idx.item())
                
                visited.update(new_nodes)
                context_nodes.extend(list(new_nodes))
                
                if not new_nodes:
                    break
            
            # Aggregate context with attention
            if context_nodes:
                context_indices = torch.tensor(context_nodes[:min(20, len(context_nodes))], device=self.device, dtype=torch.long)
                context_memories = self.nodes[context_indices]  # [num_nodes, reasoning_dim]
                
                # Ensure everything is 3D for MultiheadAttention
                if query.dim() == 1:
                    query_for_attn = query.unsqueeze(0).unsqueeze(0)  # [1, 1, reasoning_dim]
                elif query.dim() == 2:
                    query_for_attn = query.unsqueeze(1)  # [batch_size, 1, reasoning_dim]
                else:
                    query_for_attn = query
                
                context_for_attn = context_memories.unsqueeze(0)  # [1, num_nodes, reasoning_dim]
                
                if query_for_attn.size(0) != context_for_attn.size(0):
                    context_for_attn = context_for_attn.expand(query_for_attn.size(0), -1, -1)
                
                # Apply attention
                attended, attention_weights = self.graph_attention(
                    query_for_attn,
                    context_for_attn,
                    context_for_attn
                )
                
                # Process for GRU
                attended_2d = attended.squeeze(1)
                
                if query.dim() == 1:
                    query_for_gru = query
                    attended_for_gru = attended_2d.squeeze(0)
                elif query.dim() == 2:
                    query_for_gru = query.squeeze(0)
                    attended_for_gru = attended_2d.squeeze(0)
                else:
                    query_for_gru = query.squeeze(0).squeeze(0)
                    attended_for_gru = attended_2d.squeeze(0)
                
                evolved = self.memory_evolution(
                    query_for_gru,
                    attended_for_gru
                )
                
                # Identify reasoning patterns
                patterns = self.pattern_recognizer(evolved)
                pattern_probs = F.softmax(patterns, dim=-1)
                top_patterns = torch.topk(pattern_probs, k=3)
                
                metadata = {
                    'num_nodes': len(context_nodes),
                    'num_hops': hop + 1,
                    'top_similarities': top_k_scores.tolist(),
                    'graph_density': (self.adjacency_matrix[:valid_nodes, :valid_nodes] > 0).float().mean().item(),
                    'discovered_patterns': top_patterns.indices.tolist(),
                    'pattern_confidence': top_patterns.values.tolist()
                }
                
                return evolved, metadata
            
            return None, {}
            
        except Exception as e:
            if debug_mode:
                print(f"Warning: Graph retrieval failed: {e}")
                traceback.print_exc()
            return None, {}
    
    def get_trajectory_contribution_stats(self) -> Dict[int, float]:
        """Get statistics about which trajectories contribute most to memory"""
        valid_nodes = self.memory_ptr if not self.memory_filled else self.memory_size
        if valid_nodes == 0:
            return {}
        
        contributions = {}
        for traj_id in torch.unique(self.node_creators[:valid_nodes]):
            traj_memories = (self.node_creators[:valid_nodes] == traj_id).sum().item()
            avg_value = self.node_values[:valid_nodes][self.node_creators[:valid_nodes] == traj_id].mean().item()
            contributions[traj_id.item()] = {
                'count': traj_memories,
                'avg_value': avg_value,
                'percentage': traj_memories / valid_nodes
            }
        
        return contributions


# ============================================================
# TRAJECTORY LIFECYCLE MANAGER
# ============================================================

class TrajectoryLifecycleManager:
    """Manages trajectory lifecycle with specialization tracking"""
    
    def __init__(
        self,
        min_trajectories: int = 2,
        max_trajectories: int = 5,
        prune_threshold: float = 0.2,
        spawn_threshold: float = 0.8,
        diversity_bonus: float = 0.1,
        device=None
    ):
        self.min_trajectories = min_trajectories
        self.max_trajectories = max_trajectories
        self.prune_threshold = prune_threshold
        self.spawn_threshold = spawn_threshold
        self.diversity_bonus = diversity_bonus
        self.device = device
        
        self.trajectory_history = defaultdict(list)
        self.trajectory_lineage = {}
        self.specialization_tracker = TrajectorySpecializationTracker(
            max_trajectories, 256, device=device
        )
    
    def evaluate_trajectories(
        self,
        trajectories: List[TrajectoryState],
        step: int
    ) -> Tuple[List[int], List[int]]:
        """Evaluate trajectories with specialization awareness"""
        if len(trajectories) <= self.min_trajectories:
            return [], []
        
        to_prune = []
        to_spawn = []
        
        values = [t.value for t in trajectories if t.is_active]
        if not values:
            return [], []
        
        mean_value = np.mean(values)
        std_value = np.std(values) if len(values) > 1 else 0
        
        # Get diversity matrix
        diversity_matrix = self.specialization_tracker.get_diversity_matrix()
        
        for i, traj in enumerate(trajectories):
            if not traj.is_active:
                continue
            
            # Get specialization bonus
            spec_bonus = self.specialization_tracker.compute_specialization_bonus(i)
            adjusted_value = traj.value + spec_bonus
            
            # Prune with diversity awareness
            if len(trajectories) - len(to_prune) > self.min_trajectories:
                # Don't prune if trajectory is unique
                if i < len(diversity_matrix):
                    avg_diversity = diversity_matrix[i].mean().item()
                else:
                    avg_diversity = 0.5
                
                if adjusted_value < mean_value - std_value * self.prune_threshold and avg_diversity < 0.5:
                    to_prune.append(i)
                    continue
            
            # Spawn from specialized high performers
            if len(trajectories) + len(to_spawn) - len(to_prune) < self.max_trajectories:
                if adjusted_value > mean_value + std_value * self.spawn_threshold:
                    to_spawn.append(i)
        
        return to_prune, to_spawn
    
    def spawn_trajectory(
        self,
        parent: TrajectoryState,
        mutation_strength: float = 0.1
    ) -> TrajectoryState:
        """Spawn new trajectory with inherited and mutated characteristics"""
        # Inherit parent's discovered temperature preference with mutation
        new_temperature = parent.exploration_temperature * (1 + np.random.randn() * 0.1)
        new_temperature = np.clip(new_temperature, 0.5, 2.0)
        
        # Inherit risk tolerance with mutation
        new_risk = parent.risk_tolerance * (1 + np.random.randn() * 0.1)
        new_risk = np.clip(new_risk, 0.1, 0.9)
        
        # Add noise for exploration
        noise = torch.randn_like(parent.position) * mutation_strength * (1 + new_risk)
        new_position = parent.position + noise
        
        child = TrajectoryState(
            position=new_position,
            value=parent.value * 0.9,
            confidence=parent.confidence * 0.8,
            trajectory_id=parent.trajectory_id * 10 + random.randint(1, 9),
            step_count=parent.step_count,
            is_active=True,
            parent_id=parent.trajectory_id,
            exploration_temperature=new_temperature,
            risk_tolerance=new_risk
        )
        
        self.trajectory_lineage[child.trajectory_id] = parent.trajectory_id
        
        return child
    
    def _compute_diversity(self, positions: torch.Tensor) -> float:
        """Compute diversity of trajectory positions"""
        if positions.size(0) <= 1:
            return 1.0
        
        # Ensure positions are on same device
        device = positions.device
        
        # Pairwise distances
        distances = torch.cdist(positions, positions)
        
        # Mean non-diagonal distance
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=device)
        mean_distance = distances[mask].mean().item()
        
        # Normalize by dimension
        diversity = torch.sigmoid(torch.tensor(mean_distance * 2, device=device)).item()
        return diversity


# ============================================================
# COMPLEXITY ANALYZER
# ============================================================

class EnhancedComplexityAnalyzer(nn.Module):
    """Analyzes problem complexity for general reasoning"""
    
    def __init__(
        self,
        hidden_size: int,
        min_trajectories: int = 2,
        max_trajectories: int = 5,
        dtype=None
    ):
        super().__init__()
        self.min_trajectories = min_trajectories
        self.max_trajectories = max_trajectories
        self.dtype = dtype if dtype is not None else torch.bfloat16
        
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=self.dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1, dtype=self.dtype),
            nn.Sigmoid()
        )
        
        self.traj_count_net = nn.Sequential(
            nn.Linear(hidden_size + 1, 64, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(64, max_trajectories - min_trajectories + 1, dtype=self.dtype),
            nn.Softmax(dim=-1)
        )
        
        self.complexity_history = deque(maxlen=100)
    
    def forward(
        self,
        initial_state: torch.Tensor,
        return_all: bool = False
    ) -> Union[Tuple[int, float], Tuple[torch.Tensor, torch.Tensor]]:
        """Analyze complexity with proper batch handling"""
        initial_state = DimensionHelper.ensure_batch_dim(initial_state)
        batch_size = initial_state.size(0)
        
        complexity_scores = self.complexity_net(initial_state).squeeze(-1)
        
        combined = torch.cat([initial_state, complexity_scores.unsqueeze(-1)], dim=-1)
        traj_probs = self.traj_count_net(combined)
        
        if self.training:
            traj_dist = torch.distributions.Categorical(traj_probs)
            traj_offsets = traj_dist.sample()
        else:
            traj_offsets = traj_probs.argmax(dim=-1)
        
        trajectory_counts = self.min_trajectories + traj_offsets
        trajectory_counts = torch.clamp(trajectory_counts, self.min_trajectories, self.max_trajectories)
        
        self.complexity_history.extend(complexity_scores.detach().cpu().tolist())
        
        if return_all:
            return trajectory_counts, complexity_scores
        else:
            num_traj = trajectory_counts[0].item() if hasattr(trajectory_counts[0], 'item') else int(trajectory_counts[0])
            complexity = complexity_scores[0].item() if hasattr(complexity_scores[0], 'item') else float(complexity_scores[0])
            return num_traj, complexity


# ============================================================
# GENERALIZED MULTI-TRAJECTORY NAVIGATOR
# ============================================================

class GeneralizedMultiTrajectoryNavigator(nn.Module):
    """
    Navigator for general reasoning without domain-specific assumptions.
    Trajectories naturally discover their own specializations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        min_trajectories: int = 2,
        max_trajectories: int = 5,
        memory_size_per_trajectory: int = 100,
        shared_memory_size: int = 500,
        dropout_rate: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.min_trajectories = min_trajectories
        self.max_trajectories = max_trajectories
        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        
        # Core projection layers
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, dtype=self.dtype),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, reasoning_dim, dtype=self.dtype)
        )
        
        self.thought_projection = nn.Sequential(
            nn.Linear(reasoning_dim, hidden_size // 4, dtype=self.dtype),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size, dtype=self.dtype)
        )
        
        # Trajectory initialization diversity network
        self.trajectory_initializers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(reasoning_dim, reasoning_dim, dtype=self.dtype),
                nn.Tanh(),
                nn.Linear(reasoning_dim, reasoning_dim, dtype=self.dtype)
            ) if i % 2 == 0 else
            nn.Sequential(
                nn.Linear(reasoning_dim, reasoning_dim * 2, dtype=self.dtype),
                nn.ReLU(),
                nn.Linear(reasoning_dim * 2, reasoning_dim, dtype=self.dtype)
            )
            for i in range(max_trajectories)
        ])
        
        # Graph-based shared memory
        self.graph_memory = GraphMemoryBank(
            memory_size=shared_memory_size,
            reasoning_dim=reasoning_dim,
            num_heads=4,
            device=device,
            dtype=dtype
        )
        
        # Enhanced complexity analyzer
        self.complexity_analyzer = EnhancedComplexityAnalyzer(
            hidden_size, min_trajectories, max_trajectories, dtype=dtype
        )
        
        # Lifecycle manager with specialization
        self.lifecycle_manager = TrajectoryLifecycleManager(
            min_trajectories, max_trajectories, device=device
        )
        
        # Trajectory-specific heads with learnable specialization
        self.trajectory_heads = nn.ModuleList([
            nn.ModuleDict({
                'continue': nn.Sequential(
                    nn.Linear(reasoning_dim * 3, reasoning_dim, dtype=self.dtype),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(reasoning_dim, 2, dtype=self.dtype)
                ),
                'direction': nn.Linear(reasoning_dim * 3, reasoning_dim, dtype=self.dtype),
                'step_size': nn.Linear(reasoning_dim * 3, 1, dtype=self.dtype),
                'value': nn.Linear(reasoning_dim * 3, 1, dtype=self.dtype),
                'confidence': nn.Linear(reasoning_dim * 3, 1, dtype=self.dtype),
                'specialization': nn.Sequential(
                    nn.Linear(reasoning_dim * 3, reasoning_dim // 2, dtype=self.dtype),
                    nn.ReLU(),
                    nn.Linear(reasoning_dim // 2, 16, dtype=self.dtype),
                    nn.Softmax(dim=-1)
                )
            })
            for _ in range(max_trajectories)
        ])
        
        # Enhanced cross-trajectory communication
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=reasoning_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True,
            dtype=self.dtype
        )
        
        # Diversity reward network
        self.diversity_evaluator = nn.Sequential(
            nn.Linear(reasoning_dim * max_trajectories, reasoning_dim, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim, 1, dtype=self.dtype),
            nn.Sigmoid()
        )
        
        # Graph-inspired trajectory interaction
        self.trajectory_graph_encoder = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim // 2, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1, dtype=self.dtype),
            nn.Sigmoid()
        )
        
        # Ensemble aggregation with specialization awareness
        self.ensemble_gate = nn.Sequential(
            nn.Linear(reasoning_dim * (max_trajectories + 2), reasoning_dim, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(reasoning_dim, max_trajectories + 1, dtype=self.dtype),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        if device:
            self.to(device)
        
        # Ensure all submodules are on the same device
        self.device = device if device is not None else torch.device('cpu')
        print(f"Navigator initialized on device: {self.device}")
    
    def navigate_advanced(
        self,
        state: torch.Tensor,
        step_num: int = 0,
        temperature: float = 1.0,
        batch_idx: int = 0,
        return_all_info: bool = True
    ) -> Dict[str, Any]:
        """Navigate with natural trajectory specialization"""
        try:
            state = DimensionHelper.ensure_batch_dim(state, 1)
            state = state.to(self.device).to(self.dtype)
            
            # Step 1: Analyze complexity
            num_trajectories, complexity_score = self.complexity_analyzer(state, return_all=False)
            
            # Step 2: Retrieve graph context (with proper dimension handling)
            graph_context = None
            graph_metadata = {}
            
            # Only try to retrieve from graph memory if it has some content
            if self.graph_memory.total_writes > 0:
                # Project state to reasoning dimension for graph memory query
                state_projected = self.state_projection(state)
                graph_context, graph_metadata = self.graph_memory.retrieve_graph_context(
                    state_projected, num_hops=2, top_k=5, source_trajectory=0
                )
            
            # Step 3: Initialize trajectories with diverse strategies
            trajectory_states = []
            active_trajectories = []
            
            for traj_idx in range(num_trajectories):
                # Project initial state
                traj_state = self.state_projection(state)
                
                # Apply trajectory-specific initialization
                initializer = self.trajectory_initializers[traj_idx % len(self.trajectory_initializers)]
                traj_state = initializer(traj_state)
                
                # Add graph context if available
                if graph_context is not None:
                    graph_context = DimensionHelper.ensure_batch_dim(graph_context, 1)
                    weight = 0.1 + (traj_idx / num_trajectories) * 0.3
                    traj_state = (1 - weight) * traj_state + weight * graph_context
                
                # Each trajectory has its own exploration characteristics
                exploration_temp = 0.5 + (traj_idx / max(1, num_trajectories - 1)) * 1.5
                risk_tolerance = 0.2 + (traj_idx / max(1, num_trajectories - 1)) * 0.6
                
                # Add controlled noise based on trajectory's risk tolerance
                if self.training or traj_idx > 0:
                    noise_scale = 0.1 * risk_tolerance * (1 - step_num / 10)
                    noise = torch.randn_like(traj_state) * noise_scale
                    traj_state = traj_state + noise
                
                trajectory_states.append(traj_state)
                
                # Create trajectory state object
                traj_obj = TrajectoryState(
                    position=traj_state,
                    value=0.0,
                    confidence=1.0,
                    trajectory_id=traj_idx,
                    step_count=step_num,
                    is_active=True,
                    exploration_temperature=exploration_temp,
                    risk_tolerance=risk_tolerance
                )
                active_trajectories.append(traj_obj)
            
            # Step 4: Graph-inspired cross-trajectory communication
            if num_trajectories > 1:
                trajectory_graph = self._build_trajectory_graph(trajectory_states)
                
                stacked_states = DimensionHelper.safe_stack(trajectory_states, dim=1)
                
                attended_states = []
                for i in range(num_trajectories):
                    query = stacked_states[:, i:i+1, :]
                    
                    key_value_weights = trajectory_graph[i].unsqueeze(0).unsqueeze(-1)
                    weighted_kv = stacked_states * key_value_weights
                    
                    attended, attn_weights = self.cross_attention(query, weighted_kv, weighted_kv)
                    attended_states.append(attended.squeeze(1))
                
                trajectory_states = attended_states
            
            # Step 5: Lifecycle management
            if step_num > 0 and step_num % 2 == 0:
                to_prune, to_spawn = self.lifecycle_manager.evaluate_trajectories(
                    active_trajectories, step_num
                )
                
                for idx in sorted(to_prune, reverse=True):
                    if len(active_trajectories) > self.min_trajectories:
                        active_trajectories[idx].is_active = False
                
                for idx in to_spawn:
                    if len(active_trajectories) < self.max_trajectories:
                        parent = active_trajectories[idx]
                        child = self.lifecycle_manager.spawn_trajectory(parent)
                        active_trajectories.append(child)
                        
                        child_state = trajectory_states[idx] + torch.randn_like(trajectory_states[idx]) * 0.1
                        trajectory_states.append(child_state)
            
            # Step 6: Navigate each active trajectory
            trajectory_outputs = []
            related_positions = []
            
            for traj_idx, (traj_state, traj_obj) in enumerate(zip(trajectory_states, active_trajectories)):
                if not traj_obj.is_active:
                    continue
                
                traj_state = DimensionHelper.ensure_batch_dim(traj_state, 1)
                
                # Get trajectory decisions
                heads = self.trajectory_heads[traj_idx % self.max_trajectories]
                
                # Prepare input with graph context
                if graph_context is not None:
                    graph_context = DimensionHelper.ensure_batch_dim(graph_context, 1)
                    decision_input = torch.cat([traj_state, traj_state, graph_context], dim=-1)
                else:
                    padding = torch.zeros_like(traj_state)
                    decision_input = torch.cat([traj_state, traj_state, padding], dim=-1)
                
                # Use trajectory's own temperature
                traj_temp = temperature * traj_obj.exploration_temperature
                
                # Make decisions
                continue_logits = heads['continue'](decision_input) / traj_temp
                continue_probs = F.softmax(continue_logits, dim=-1)
                
                if self.training:
                    continue_dist = torch.distributions.Categorical(continue_probs)
                    continue_action = continue_dist.sample()
                else:
                    continue_action = continue_probs.argmax(dim=-1)
                
                # Navigation with trajectory-specific characteristics
                direction = F.normalize(heads['direction'](decision_input), p=2, dim=-1)
                step_size = torch.sigmoid(heads['step_size'](decision_input)) * 2.0 * traj_obj.risk_tolerance
                value = heads['value'](decision_input)
                confidence = torch.sigmoid(heads['confidence'](decision_input))
                
                # Discover specialization
                specialization = heads['specialization'](decision_input)
                
                # Move in reasoning space
                next_position = traj_state + step_size * direction
                latent_thought = self.thought_projection(next_position)
                
                # Update specialization tracker
                step_size_scalar = step_size.item() if hasattr(step_size, 'item') else float(step_size.view(-1)[0])
                value_scalar = value.item() if hasattr(value, 'item') else float(value.view(-1)[0])
                confidence_scalar = confidence.item() if hasattr(confidence, 'item') else float(confidence.view(-1)[0])
                
                self.lifecycle_manager.specialization_tracker.update_trajectory_signature(
                    traj_idx,
                    step_size_scalar,
                    direction.squeeze(0),
                    value_scalar,
                    confidence_scalar,
                    traj_state.squeeze(0)
                )
                
                # Update trajectory object
                traj_obj.position = next_position
                traj_obj.value = value_scalar
                traj_obj.confidence = confidence_scalar
                traj_obj.step_count = step_num
                
                related_positions.append(next_position)
                
                trajectory_outputs.append({
                    'thought': latent_thought,
                    'position': next_position,
                    'stop_vote': continue_action == 1,
                    'continue_prob': continue_probs[:, 0] if continue_probs.dim() > 1 else continue_probs[0],
                    'value': value,
                    'confidence': confidence,
                    'trajectory_idx': traj_idx,
                    'trajectory_obj': traj_obj,
                    'specialization': specialization
                })
            
            # Step 7: Compute diversity reward
            if trajectory_outputs and len(trajectory_outputs) > 1:
                all_positions = [out['position'].squeeze(0) for out in trajectory_outputs]
                while len(all_positions) < self.max_trajectories:
                    all_positions.append(torch.zeros_like(all_positions[0]))
                
                concat_positions = torch.cat(all_positions[:self.max_trajectories])
                diversity_score = self.diversity_evaluator(concat_positions.unsqueeze(0)).item()
            else:
                diversity_score = 0.0
            
            # Step 8: Write best trajectories to graph memory
            if trajectory_outputs:
                values = [out['value'].view(-1)[0] for out in trajectory_outputs]
                
                # Apply diversity bonus to values
                adjusted_values = []
                for i, v in enumerate(values):
                    spec_bonus = self.lifecycle_manager.specialization_tracker.compute_specialization_bonus(i)
                    adjusted_values.append(v.item() + spec_bonus + diversity_score * 0.1)
                
                best_idx = np.argmax(adjusted_values)
                best_output = trajectory_outputs[best_idx]
                
                node_idx = self.graph_memory.write(
                    best_output['position'],
                    best_output['value'],
                    best_output['trajectory_idx'],
                    related_positions
                )
            
            # Step 9: Ensemble aggregation
            ensemble_thought, ensemble_stop = self._aggregate_trajectories(
                trajectory_outputs, graph_context, num_trajectories
            )
            
            # Get contribution stats
            contribution_stats = self.graph_memory.get_trajectory_contribution_stats()
            
            result = {
                'ensemble_thought': ensemble_thought,
                'ensemble_stop': ensemble_stop,
                'trajectory_outputs': trajectory_outputs,
                'num_trajectories': num_trajectories,
                'num_active': sum(1 for t in active_trajectories if t.is_active),
                'complexity_score': complexity_score,
                'diversity_score': diversity_score,
                'graph_metadata': graph_metadata,
                'contribution_stats': contribution_stats,
                'specialization_info': {
                    i: {
                        'avg_step_size': np.mean(list(sig['avg_step_size'])) if sig['avg_step_size'] else 0,
                        'avg_confidence': np.mean(list(sig['confidence_profile'])) if sig['confidence_profile'] else 0,
                        'success_rate': len(sig['success_contexts']) / max(1, len(sig['success_contexts']) + len(sig['failure_contexts']))
                    }
                    for i, sig in self.lifecycle_manager.specialization_tracker.trajectory_signatures.items()
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Navigation error at step {step_num}: {e}")
            traceback.print_exc()
            
            fallback_thought = self.thought_projection(
                self.state_projection(state)
            )
            
            return {
                'ensemble_thought': fallback_thought,
                'ensemble_stop': torch.tensor([True], device=self.device),
                'trajectory_outputs': [],
                'num_trajectories': 1,
                'num_active': 1,
                'complexity_score': 0.5,
                'diversity_score': 0.0,
                'error': str(e)
            }
    
    def _build_trajectory_graph(self, trajectory_states: List[torch.Tensor]) -> torch.Tensor:
        """Build graph relationships between trajectories"""
        num_traj = len(trajectory_states)
        
        states = [DimensionHelper.ensure_batch_dim(s, 1).squeeze(0) for s in trajectory_states]
        
        graph = torch.zeros(num_traj, num_traj, device=self.device, dtype=self.dtype)
        
        for i in range(num_traj):
            for j in range(i + 1, num_traj):
                combined = torch.cat([states[i], states[j]])
                strength = self.trajectory_graph_encoder(combined).item()
                
                graph[i, j] = strength
                graph[j, i] = strength
        
        graph.fill_diagonal_(1.0)
        graph = F.softmax(graph, dim=-1)
        
        return graph
    
    def _aggregate_trajectories(
        self,
        trajectory_outputs: List[Dict],
        graph_context: Optional[torch.Tensor],
        num_trajectories: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate trajectory outputs with specialization awareness"""
        if not trajectory_outputs:
            return (
                torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype),
                torch.tensor([True], device=self.device)
            )
        
        thoughts = []
        for out in trajectory_outputs:
            thought = out['thought']
            thought = DimensionHelper.ensure_batch_dim(thought, 1).squeeze(0)
            thoughts.append(thought)
        
        while len(thoughts) < self.max_trajectories:
            thoughts.append(torch.zeros_like(thoughts[0]))
        
        thoughts_tensor = DimensionHelper.safe_stack(thoughts[:self.max_trajectories])
        
        all_states = [out['position'] for out in trajectory_outputs]
        all_states = [DimensionHelper.ensure_batch_dim(s, 1).squeeze(0) for s in all_states]
        
        while len(all_states) < self.max_trajectories:
            all_states.append(torch.zeros_like(all_states[0]))
        
        concat_states = torch.cat(all_states[:self.max_trajectories])
        
        if graph_context is not None:
            graph_context = DimensionHelper.ensure_batch_dim(graph_context, 1).squeeze(0)
            ensemble_input = torch.cat([concat_states, graph_context, graph_context])
        else:
            padding = torch.zeros(self.reasoning_dim * 2, device=self.device, dtype=self.dtype)
            ensemble_input = torch.cat([concat_states, padding])
        
        ensemble_weights = self.ensemble_gate(ensemble_input.unsqueeze(0))
        
        if thoughts_tensor.dim() == 2:
            thoughts_tensor = thoughts_tensor.unsqueeze(0)
        
        weighted_thoughts = thoughts_tensor * ensemble_weights[:, :self.max_trajectories].unsqueeze(-1)
        ensemble_thought = weighted_thoughts.sum(dim=1).squeeze(0)
        
        stop_votes = []
        for out in trajectory_outputs:
            stop_vote = out['stop_vote']
            if isinstance(stop_vote, bool):
                stop_vote = torch.tensor([float(stop_vote)], device=self.device, dtype=self.dtype)
            elif hasattr(stop_vote, 'item'):
                stop_vote = torch.tensor([stop_vote.item()], device=self.device, dtype=self.dtype)
            stop_votes.append(stop_vote[0] if stop_vote.numel() > 0 else stop_vote)
        
        while len(stop_votes) < self.max_trajectories:
            stop_votes.append(torch.tensor(0.0, device=self.device, dtype=self.dtype))
        
        stop_tensor = torch.stack(stop_votes[:self.max_trajectories])
        weighted_stop = (stop_tensor * ensemble_weights[0, :self.max_trajectories]).sum() > 0.5
        
        return ensemble_thought, weighted_stop


# ============================================================
# COMPLETE MODEL
# ============================================================

class GeneralizedMultiTrajectoryCoconut(nn.Module):
    """Complete model for general reasoning with natural trajectory specialization"""
    
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        hidden_size: int = 4096,
        reasoning_dim: int = 256,
        min_trajectories: int = 2,
        max_trajectories: int = 5,
        max_latent_steps: int = 6,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.max_latent_steps = max_latent_steps
        
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        
        self.navigator = GeneralizedMultiTrajectoryNavigator(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
            min_trajectories=min_trajectories,
            max_trajectories=max_trajectories,
            dropout_rate=dropout_rate,
            device=device,
            dtype=dtype
        )
        
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            self.base_model.config.use_cache = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trajectory_info: bool = True,
        temperature: float = 1.0,
        force_reasoning: bool = True  # Add flag to ensure reasoning happens
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with natural trajectory specialization"""
        try:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            
            batch_trajectory_info = []
            batch_ensemble_sequences = []
            
            # Get initial states for all batch items
            with torch.no_grad():
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                last_hidden = outputs.hidden_states[-1]
                initial_states = last_hidden.mean(dim=1)
            
            # Navigate for each batch item
            for b in range(batch_size):
                current_state = initial_states[b]
                
                trajectory_info = {
                    'num_trajectories': [],
                    'num_active': [],
                    'complexity_scores': [],
                    'diversity_scores': [],
                    'specialization_info': [],
                    'ensemble_thoughts': [],
                    'stop_patterns': []
                }
                
                ensemble_sequence = []
                actual_steps = 0
                
                # Navigate through reasoning steps
                for step in range(self.max_latent_steps):
                    nav_output = self.navigator.navigate_advanced(
                        current_state,
                        step_num=step,
                        temperature=temperature,
                        batch_idx=b
                    )
                    
                    actual_steps += 1
                    
                    # Store trajectory info
                    trajectory_info['num_trajectories'].append(nav_output.get('num_trajectories', 0))
                    trajectory_info['num_active'].append(nav_output.get('num_active', 0))
                    trajectory_info['complexity_scores'].append(nav_output.get('complexity_score', 0))
                    trajectory_info['diversity_scores'].append(nav_output.get('diversity_score', 0))
                    trajectory_info['specialization_info'].append(nav_output.get('specialization_info', {}))
                    
                    # Check stop condition
                    stop_value = nav_output['ensemble_stop']
                    if hasattr(stop_value, 'item'):
                        should_stop = stop_value.item()
                    elif isinstance(stop_value, bool):
                        should_stop = stop_value
                    else:
                        should_stop = bool(stop_value)
                    
                    trajectory_info['stop_patterns'].append(should_stop)
                    
                    # Force at least 2 steps if force_reasoning is True
                    if force_reasoning and step < 1:
                        should_stop = False
                    
                    # Don't allow stopping after max steps either (must use all steps in training)
                    if force_reasoning and step >= self.max_latent_steps - 1:
                        should_stop = True
                    
                    if should_stop:
                        break
                    
                    # Use ensemble thought for next step
                    ensemble_thought = nav_output['ensemble_thought']
                    ensemble_thought = DimensionHelper.ensure_batch_dim(ensemble_thought, 1).squeeze(0)
                    
                    ensemble_sequence.append(ensemble_thought)
                    trajectory_info['ensemble_thoughts'].append(ensemble_thought)
                    
                    current_state = ensemble_thought
                
                # Store actual steps taken
                trajectory_info['actual_steps'] = actual_steps
                
                batch_trajectory_info.append(trajectory_info)
                batch_ensemble_sequences.append(ensemble_sequence)
            
            # Process through base model with proper batching
            max_latent_len = max(len(seq) for seq in batch_ensemble_sequences) if batch_ensemble_sequences else 0
            
            # Ensure we actually did some reasoning
            if max_latent_len == 0 and force_reasoning:
                print(f"Warning: No reasoning steps were taken! Force adding one step.")
                # Force at least one reasoning step
                for b in range(batch_size):
                    dummy_thought = torch.zeros(self.hidden_size, device=device, dtype=inputs_embeds.dtype)
                    batch_ensemble_sequences[b] = [dummy_thought]
                max_latent_len = 1
            
            if max_latent_len > 0:
                # Adjust labels
                if labels is not None:
                    batch_size, seq_len = labels.shape
                    new_labels = torch.full(
                        (batch_size, seq_len + max_latent_len),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    new_labels[:, 0] = labels[:, 0]
                    new_labels[:, max_latent_len + 1:] = labels[:, 1:]
                    labels = new_labels
                
                # Pad sequences properly
                padded_sequences = []
                for seq in batch_ensemble_sequences:
                    if len(seq) == 0:
                        padding = [torch.zeros(self.hidden_size, device=device, dtype=inputs_embeds.dtype) 
                                  for _ in range(max_latent_len)]
                        padded_seq = padding
                    else:
                        seq = [DimensionHelper.ensure_batch_dim(s, 1).squeeze(0) for s in seq]
                        
                        if len(seq) < max_latent_len:
                            padding = [torch.zeros_like(seq[0]) for _ in range(max_latent_len - len(seq))]
                            padded_seq = seq + padding
                        else:
                            padded_seq = seq
                    
                    stacked_seq = DimensionHelper.safe_stack(padded_seq, dim=0)
                    padded_sequences.append(stacked_seq)
                
                latent_embeds = torch.stack(padded_sequences, dim=0)
                
                # Create enhanced embeddings
                enhanced_embeds = torch.cat([
                    inputs_embeds[:, :1, :],
                    latent_embeds,
                    inputs_embeds[:, 1:, :]
                ], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    latent_attention = torch.ones(
                        batch_size, max_latent_len,
                        dtype=attention_mask.dtype,
                        device=device
                    )
                    enhanced_mask = torch.cat([
                        attention_mask[:, :1],
                        latent_attention,
                        attention_mask[:, 1:]
                    ], dim=1)
                else:
                    enhanced_mask = None
                
                outputs = self.base_model(
                    inputs_embeds=enhanced_embeds,
                    attention_mask=enhanced_mask,
                    labels=labels
                )
            else:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            result = {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
            
            if return_trajectory_info and batch_trajectory_info:
                result['trajectory_info'] = batch_trajectory_info[0] if batch_size == 1 else batch_trajectory_info
                result['avg_trajectories'] = np.mean([
                    info['num_trajectories'][0] if info['num_trajectories'] else 0 
                    for info in batch_trajectory_info
                ])
                result['avg_complexity'] = np.mean([
                    info['complexity_scores'][0] if info['complexity_scores'] else 0 
                    for info in batch_trajectory_info
                ])
                result['avg_diversity'] = np.mean([
                    np.mean(info['diversity_scores']) if info['diversity_scores'] else 0 
                    for info in batch_trajectory_info
                ])
                result['trajectory_length'] = np.mean([len(seq) for seq in batch_ensemble_sequences])
            
            return result
            
        except Exception as e:
            print(f"Forward pass error: {e}")
            traceback.print_exc()
            
            return {
                'loss': torch.tensor(0.0, device=input_ids.device, requires_grad=True),
                'logits': torch.zeros(
                    input_ids.shape[0], input_ids.shape[1], 50000,
                    device=input_ids.device
                ),
                'error': str(e)
            }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_generalized_coconut(debug_mode=False):
    """
    Complete training function for Generalized Multi-Trajectory COCONUT.
    Trains trajectories to naturally discover their own specializations.
    
    Args:
        debug_mode: If True, run with minimal samples for debugging
    """
    
    # Configuration
    num_epochs = 1 if debug_mode else 3
    batch_size = 1 if debug_mode else 2
    max_batch_size = 6
    min_batch_size = 1
    target_effective_batch = 32
    gradient_accumulation_steps = 1 if debug_mode else 16
    max_length = 256 if debug_mode else 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Force slower, more careful training for quality
    if not debug_mode:
        print("\n⚠️ Note: Training with multi-trajectory reasoning is computationally intensive.")
        print("   Expected speed: 0.5-2.0 it/s depending on trajectory count and steps.")
        print("   If training is too fast (>2 it/s), reasoning may be skipped!")
    
    # Checkpointing
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Metrics tracking
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'trajectory_specialization': [],
        'diversity_scores': [],
        'memory_usage': []
    }
    
    # Model configuration - ensure proper reasoning
    MIN_TRAJECTORIES = 2
    MAX_TRAJECTORIES = 4 if not debug_mode else 3  # Reduced for memory but ensure multiple
    MAX_LATENT_STEPS = 4 if not debug_mode else 3  # Ensure multiple steps
    
    print(f"\n🎯 Reasoning Configuration:")
    print(f"   • Min trajectories: {MIN_TRAJECTORIES}")
    print(f"   • Max trajectories: {MAX_TRAJECTORIES}")
    print(f"   • Max reasoning steps: {MAX_LATENT_STEPS}")
    print(f"   • This should result in 2-4 trajectories and 2-4 steps per problem")
    
    # Temperature schedule
    INITIAL_TEMP = 1.5
    FINAL_TEMP = 0.5
    
    # Learning rates
    base_learning_rate = 2.5e-6
    navigator_lr = 5e-6
    
    print("\n" + "="*70)
    print("GENERALIZED MULTI-TRAJECTORY COCONUT TRAINING")
    print("="*70)
    print("\n🎯 Key Features:")
    print("  • Trajectories naturally discover specializations")
    print("  • No pre-specified reasoning strategies")
    print("  • Graph memory enables emergent collaboration")
    print("  • Diversity rewards encourage exploration")
    print("  • Adaptive lifecycle management")
    print(f"\n📱 Device: {device}")
    print("="*70)
    
    print(f"\n🎯 Configuration:")
    print(f"   • Epochs: {num_epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   • Max sequence length: {max_length}")
    print(f"   • Trajectories: {MIN_TRAJECTORIES}-{MAX_TRAJECTORIES}")
    print(f"   • Device: {device}")
    
    if debug_mode:
        print("\n🔍 DEBUG MODE ACTIVE - Processing minimal samples")
    
    # Memory monitoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n📊 GPU Memory: {initial_memory:.1f}GB used / {total_memory:.1f}GB total")
    
    # Load tokenizer and model
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    print("🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    
    print("🧠 Creating Generalized Multi-Trajectory COCONUT model...")
    model = GeneralizedMultiTrajectoryCoconut(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=256,
        min_trajectories=MIN_TRAJECTORIES,
        max_trajectories=MAX_TRAJECTORIES,
        max_latent_steps=MAX_LATENT_STEPS,
        dropout_rate=0.1
    )
    
    # Freeze early layers
    print("❄️ Freezing early layers (keeping last 4 layers trainable)...")
    for i, layer in enumerate(model.base_model.model.layers):
        if i < 28:  # Freeze first 28 of 32 layers
            for param in layer.parameters():
                param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable: {trainable_params/1e6:.2f}M / {total_params/1e9:.2f}B")
    
    # Load dataset
    print("\n📊 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    val_dataset = load_dataset("gsm8k", "main", split="test")
    
    model.to(device)
    
    # Test a forward pass
    print("\n🔧 Testing forward pass...")
    try:
        test_item = dataset[0]
        test_prompt = f"Question: {test_item['question']}\nLet's solve this step by step.\n\nSolution: {test_item['answer']}"
        test_inputs = tokenizer(test_prompt, return_tensors='pt', truncation=True, max_length=max_length).to(device)
        
        test_start = time.time()
        with torch.no_grad():
            test_output = model(
                input_ids=test_inputs['input_ids'],
                attention_mask=test_inputs['attention_mask'],
                labels=test_inputs['input_ids'],
                temperature=INITIAL_TEMP,
                force_reasoning=True
            )
        test_time = time.time() - test_start
        
        if 'error' in test_output:
            print(f"⚠️ Forward pass test failed: {test_output['error']}")
        else:
            print("✅ Forward pass test successful!")
            print(f"   • Trajectories spawned: {test_output.get('avg_trajectories', 0):.1f}")
            print(f"   • Reasoning steps: {test_output.get('trajectory_length', 0):.1f}")
            print(f"   • Forward pass time: {test_time:.2f}s")
            
            # Check if reasoning is actually happening
            if test_output.get('trajectory_length', 0) == 0:
                print("   ⚠️ WARNING: No reasoning steps detected! Check navigator.")
            
            expected_speed = 1.0 / test_time  # items per second
            print(f"   • Expected training speed: ~{expected_speed:.1f} it/s")
            
            if expected_speed > 3.0:
                print("   ⚠️ Model is too fast! Reasoning may be skipped.")
                print("      Check that navigator.navigate_advanced() is being called.")
        
        # Clean up test
        del test_output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"⚠️ Forward pass test failed with error: {e}")
        print("Continuing anyway...")
    
    # Optimizer setup
    navigator_params = list(model.navigator.parameters())
    navigator_param_ids = {id(p) for p in navigator_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in navigator_param_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': navigator_params, 'lr': navigator_lr, 'weight_decay': 0.001},
        {'params': base_params, 'lr': base_learning_rate, 'weight_decay': 0.01}
    ], betas=(0.9, 0.95))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    print("\n🚀 Starting training!")
    print(f"   • Batch size: {batch_size} × {gradient_accumulation_steps} accumulation")
    print(f"   • Trajectories: {MIN_TRAJECTORIES}-{MAX_TRAJECTORIES} (naturally specialized)")
    print(f"   • Max reasoning steps: {MAX_LATENT_STEPS}")
    print(f"   • Temperature: {INITIAL_TEMP} → {FINAL_TEMP}")
    print("="*70)
    
    global_step = 0
    best_val_accuracy = 0.0
    total_successful_samples = 0
    total_failed_samples = 0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Temperature annealing
        temperature = INITIAL_TEMP - (INITIAL_TEMP - FINAL_TEMP) * (epoch / num_epochs)
        print(f"   Temperature: {temperature:.2f}")
        
        epoch_losses = []
        epoch_correct = []
        epoch_diversity = []
        epoch_successful = 0
        epoch_failed = 0
        epoch_skipped = 0
        epoch_trajectory_counts = []
        epoch_step_counts = []
        specialization_patterns = defaultdict(list)
        
        # Track accuracy by number of trajectories and steps
        accuracy_by_trajectories = defaultdict(list)
        accuracy_by_steps = defaultdict(list)
        
        dataset_size = min(len(dataset), 10 if debug_mode else 500)
        num_batches = dataset_size // batch_size
        
        if debug_mode:
            print(f"\n🔍 DEBUG MODE: Processing only {dataset_size} samples")
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        # Debug: Track what's happening
        debug_first_batch = True
        
        for batch_idx in progress_bar:
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min(batch_start_idx + batch_size, dataset_size)
            batch_items = [dataset[i] for i in range(batch_start_idx, batch_end_idx)]
            
            batch_losses = []
            batch_correct = []
            batch_errors = 0
            batch_timing = {'forward': 0, 'backward': 0, 'total': 0}
            batch_start_time = time.time()
            
            for item_idx, item in enumerate(batch_items):
                question = item['question']
                answer_text = item['answer']
                
                train_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution: {answer_text}"
                inference_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
                
                try:
                    # Timing
                    item_start_time = time.time()
                    
                    # Training forward pass
                    inputs = tokenizer(
                        train_prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length,
                        padding=False
                    ).to(device)
                    
                    # Forward pass with reasoning
                    forward_start = time.time()
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids'],
                        temperature=temperature,
                        return_trajectory_info=True,
                        force_reasoning=True  # Ensure reasoning happens
                    )
                    forward_time = time.time() - forward_start
                    batch_timing['forward'] += forward_time
                    
                    # Check if we got valid outputs
                    if outputs is None or 'loss' not in outputs:
                        print(f"\n⚠️ Invalid outputs in batch {batch_idx}, item {item_idx}")
                        epoch_skipped += 1
                        continue
                    
                    # Check if loss is valid
                    if outputs['loss'] is None or not torch.isfinite(outputs['loss']):
                        print(f"\n⚠️ Invalid loss in batch {batch_idx}, item {item_idx}: {outputs['loss']}")
                        epoch_skipped += 1
                        continue
                    
                    # Extract trajectory metrics
                    num_trajectories_used = outputs.get('avg_trajectories', 0)
                    trajectory_length = outputs.get('trajectory_length', 0)
                    diversity_score = outputs.get('avg_diversity', 0)
                    
                    # Verify reasoning is happening
                    if trajectory_length == 0:
                        print(f"\n⚠️ Warning: No reasoning steps in sample {item_idx}!")
                    
                    # Track detailed trajectory info
                    if 'trajectory_info' in outputs and outputs['trajectory_info']:
                        traj_info = outputs['trajectory_info']
                        if isinstance(traj_info, list):
                            traj_info = traj_info[0]
                        
                        # Track number of trajectories and steps
                        if 'num_trajectories' in traj_info and traj_info['num_trajectories']:
                            actual_trajectories = traj_info['num_trajectories'][0]
                            epoch_trajectory_counts.append(actual_trajectories)
                        
                        # Track actual reasoning steps taken
                        if 'stop_patterns' in traj_info:
                            actual_steps = len(traj_info['stop_patterns'])
                            epoch_step_counts.append(actual_steps)
                        elif 'actual_steps' in traj_info:
                            actual_steps = traj_info['actual_steps']
                            epoch_step_counts.append(actual_steps)
                        else:
                            actual_steps = 0
                        
                        if 'specialization_info' in traj_info:
                            for step_spec in traj_info['specialization_info']:
                                for traj_id, spec in step_spec.items():
                                    specialization_patterns[traj_id].append(spec)
                    
                    # Check answer correctness (only for some samples to save time)
                    is_correct = False
                    if item_idx == 0 or epoch_successful % 10 == 0:  # Check every 10th sample
                        try:
                            # Generate answer using the model
                            with torch.no_grad():
                                inference_inputs = tokenizer(
                                    inference_prompt,
                                    return_tensors='pt',
                                    truncation=True,
                                    max_length=max_length // 2
                                ).to(device)
                                
                                # Use the base model for generation
                                generated = model.base_model.generate(
                                    inference_inputs['input_ids'],
                                    attention_mask=inference_inputs['attention_mask'],
                                    max_new_tokens=50,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id
                                )
                                
                                response = tokenizer.decode(
                                    generated[0][inference_inputs['input_ids'].shape[1]:], 
                                    skip_special_tokens=True
                                )
                                
                                # Extract numbers from response and answer
                                pred_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', response)
                                true_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
                                
                                if pred_nums and true_nums:
                                    pred_answer = pred_nums[-1].replace(',', '')
                                    true_answer = true_nums[-1].replace(',', '')
                                    
                                    try:
                                        pred_val = float(pred_answer)
                                        true_val = float(true_answer)
                                        is_correct = abs(pred_val - true_val) < 0.01
                                    except:
                                        pass
                            
                            batch_correct.append(is_correct)
                            epoch_correct.append(is_correct)
                        except:
                            pass
                    
                    # Track accuracy by trajectories and steps (even if we didn't check this sample)
                    if num_trajectories_used > 0 and len(batch_correct) > 0:
                        # Use last known accuracy for tracking
                        last_accuracy = batch_correct[-1] if batch_correct else False
                        accuracy_by_trajectories[int(num_trajectories_used)].append(last_accuracy)
                        
                    if trajectory_length > 0 and len(batch_correct) > 0:
                        last_accuracy = batch_correct[-1] if batch_correct else False
                        accuracy_by_steps[int(trajectory_length)].append(last_accuracy)
                    
                    # Track metrics
                    epoch_diversity.append(diversity_score)
                    
                    # Compute loss with diversity reward
                    loss = outputs['loss']
                    if diversity_score > 0:
                        diversity_reward = diversity_score * 0.1
                        loss = loss - diversity_reward
                    
                    loss = loss / gradient_accumulation_steps
                    
                    # Debug first successful sample
                    if debug_first_batch and epoch_successful == 0:
                        print(f"\n📝 First successful sample debug:")
                        print(f"   • Input shape: {inputs['input_ids'].shape}")
                        print(f"   • Loss value: {loss.item():.4f}")
                        print(f"   • Trajectories: {num_trajectories_used}")
                        print(f"   • Steps taken: {trajectory_length}")
                        print(f"   • Diversity: {diversity_score:.3f}")
                        print(f"   • Correct: {is_correct}")
                        print(f"   • Has gradient: {loss.requires_grad}")
                        debug_first_batch = False
                    
                    # Backward pass
                    backward_start = time.time()
                    loss.backward()
                    backward_time = time.time() - backward_start
                    batch_timing['backward'] += backward_time
                    
                    batch_losses.append(loss.item())
                    epoch_losses.append(loss.item())
                    epoch_successful += 1
                    
                    # Debug: Show reasoning is happening
                    if epoch_successful <= 3:
                        check_mark = "✓" if is_correct else "✗" if len(batch_correct) > 0 else "?"
                        print(f"\n  Sample {epoch_successful}: {int(num_trajectories_used)} traj, {int(trajectory_length)} steps, acc={check_mark}, t={forward_time:.2f}s")
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"\n⚠️ OOM in batch {batch_idx}, item {item_idx}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    batch_errors += 1
                    epoch_failed += 1
                    break
                    
                except RuntimeError as e:
                    if "device" in str(e).lower():
                        batch_errors += 1
                        epoch_failed += 1
                        if batch_errors == 1:
                            print(f"\n⚠️ Device error in batch {batch_idx}: {e}")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        continue
                    else:
                        raise e
                    
                except Exception as e:
                    batch_errors += 1
                    epoch_failed += 1
                    if batch_errors == 1:
                        print(f"\n⚠️ Error in batch {batch_idx}: {type(e).__name__}: {e}")
                    continue
            
            # Skip gradient update if too many errors
            if batch_errors >= len(batch_items):
                print(f"\n⚠️ Batch {batch_idx} skipped due to errors")
                optimizer.zero_grad()
                continue
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Batch timing
            batch_timing['total'] = time.time() - batch_start_time
            
            # Update progress bar with detailed info
            if batch_losses or batch_errors > 0:
                postfix = {}
                
                if batch_losses:
                    postfix['loss'] = f"{np.mean(batch_losses):.3f}"
                
                # Show accuracy (if we have any checks)
                if epoch_correct:
                    postfix['acc'] = f"{np.mean(epoch_correct):.1%}"
                elif batch_correct:
                    postfix['acc'] = f"{np.mean(batch_correct):.1%}"
                
                # Show average trajectories and steps
                if epoch_trajectory_counts:
                    postfix['traj'] = f"{np.mean(epoch_trajectory_counts[-10:]):.1f}"
                
                if epoch_step_counts:
                    postfix['steps'] = f"{np.mean(epoch_step_counts[-10:]):.1f}"
                
                if epoch_diversity:
                    postfix['div'] = f"{np.mean(epoch_diversity[-10:]):.2f}"
                
                # Show counts
                postfix['ok'] = epoch_successful
                
                # Show timing (only occasionally to avoid clutter)
                if batch_idx % 10 == 0 and batch_timing['forward'] > 0:
                    postfix['t'] = f"{batch_timing['total']:.1f}s"
                
                # Memory usage
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    current_mem = torch.cuda.memory_allocated() / 1024**3
                    postfix['mem'] = f"{current_mem:.1f}G"
                    metrics['memory_usage'].append(current_mem)
                
                progress_bar.set_postfix(postfix)
        
        # Epoch summary
        total_successful_samples += epoch_successful
        total_failed_samples += epoch_failed
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  • Loss: {np.mean(epoch_losses):.4f}" if epoch_losses else "  • Loss: N/A")
        print(f"  • Accuracy: {np.mean(epoch_correct):.2%} ({sum(epoch_correct)}/{len(epoch_correct)})" if epoch_correct else "  • Accuracy: N/A")
        print(f"  • Successful samples: {epoch_successful}")
        print(f"  • Failed samples: {epoch_failed}")
        print(f"  • Skipped samples: {epoch_skipped}")
        print(f"  • Success rate: {epoch_successful/(epoch_successful + epoch_failed + epoch_skipped)*100:.1f}%" if (epoch_successful + epoch_failed + epoch_skipped) > 0 else "N/A")
        print(f"  • Avg diversity: {np.mean(epoch_diversity):.3f}" if epoch_diversity else "  • Avg diversity: N/A")
        print(f"  • Avg trajectories: {np.mean(epoch_trajectory_counts):.2f}" if epoch_trajectory_counts else "  • Avg trajectories: N/A")
        print(f"  • Avg steps: {np.mean(epoch_step_counts):.2f}" if epoch_step_counts else "  • Avg steps: N/A")
        
        # Show reasoning verification
        if epoch_trajectory_counts or epoch_step_counts:
            print(f"\n🔍 Reasoning Verification:")
            if epoch_trajectory_counts:
                print(f"  • Trajectories used: min={min(epoch_trajectory_counts)}, max={max(epoch_trajectory_counts)}, avg={np.mean(epoch_trajectory_counts):.1f}")
            if epoch_step_counts:
                print(f"  • Steps taken: min={min(epoch_step_counts)}, max={max(epoch_step_counts)}, avg={np.mean(epoch_step_counts):.1f}")
            
            # Check if reasoning is actually happening
            if epoch_step_counts and np.mean(epoch_step_counts) < 1.5:
                print("  ⚠️ WARNING: Average steps < 1.5 - reasoning may not be working properly!")
            if epoch_trajectory_counts and np.mean(epoch_trajectory_counts) < 1.5:
                print("  ⚠️ WARNING: Average trajectories < 1.5 - multi-trajectory not working!")
        
        # Distribution of accuracy by number of trajectories
        if accuracy_by_trajectories:
            print(f"\n📈 Accuracy by Number of Trajectories:")
            for num_traj in sorted(accuracy_by_trajectories.keys()):
                acc_list = accuracy_by_trajectories[num_traj]
                if acc_list:
                    accuracy = np.mean(acc_list)
                    count = len(acc_list)
                    bar = '█' * int(accuracy * 20)
                    print(f"  {num_traj} trajectories: {bar:<20} {accuracy:.1%} ({sum(acc_list)}/{count} samples)")
        
        # Distribution of accuracy by number of steps
        if accuracy_by_steps:
            print(f"\n📈 Accuracy by Number of Steps:")
            for num_steps in sorted(accuracy_by_steps.keys()):
                acc_list = accuracy_by_steps[num_steps]
                if acc_list:
                    accuracy = np.mean(acc_list)
                    count = len(acc_list)
                    bar = '█' * int(accuracy * 20)
                    print(f"  {num_steps} steps: {bar:<20} {accuracy:.1%} ({sum(acc_list)}/{count} samples)")
        
        # Distribution of trajectory counts
        if epoch_trajectory_counts:
            print(f"\n📊 Trajectory Count Distribution:")
            traj_counter = Counter(epoch_trajectory_counts)
            total_samples = sum(traj_counter.values())
            for num_traj in sorted(traj_counter.keys()):
                count = traj_counter[num_traj]
                pct = count / total_samples * 100
                bar = '█' * int(pct / 2)
                print(f"  {num_traj} trajectories: {bar:<25} {count} samples ({pct:.1f}%)")
        
        # Distribution of step counts
        if epoch_step_counts:
            print(f"\n📊 Step Count Distribution:")
            step_counter = Counter(epoch_step_counts)
            total_samples = sum(step_counter.values())
            for num_steps in sorted(step_counter.keys()):
                count = step_counter[num_steps]
                pct = count / total_samples * 100
                bar = '█' * int(pct / 2)
                print(f"  {num_steps} steps: {bar:<25} {count} samples ({pct:.1f}%)")
        
        # Show if data was actually processed
        if epoch_successful == 0:
            print("\n  ⚠️ WARNING: No samples were successfully processed this epoch!")
            print("  This indicates a serious issue with the training loop.")
            print("  Possible causes:")
            print("    - Memory issues (try reducing batch_size)")
            print("    - Model configuration issues")
            print("    - Data loading problems")
            print("\n  Run with --test flag to debug a single sample:")
            print("    python script.py --test")
        
        # Analyze specialization patterns
        if specialization_patterns:
            print(f"\n🔬 Trajectory Specialization Analysis:")
            for traj_id, patterns in list(specialization_patterns.items())[:3]:
                if patterns:
                    avg_step = np.mean([p.get('avg_step_size', 0) for p in patterns])
                    avg_conf = np.mean([p.get('avg_confidence', 0) for p in patterns])
                    success_rate = np.mean([p.get('success_rate', 0) for p in patterns])
                    
                    print(f"  Trajectory {traj_id}:")
                    print(f"    • Avg step size: {avg_step:.3f}")
                    print(f"    • Avg confidence: {avg_conf:.3f}")
                    print(f"    • Success rate: {success_rate:.2%}")
        
        # Skip validation in debug mode
        if not debug_mode:
            # Validation
            print(f"\n🔍 Running validation...")
            val_correct = 0
            val_total = 30
            
            model.eval()
            for i in tqdm(range(val_total), desc="Validating"):
                item = val_dataset[i]
                question = item['question']
                answer_text = item['answer']
                
                prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
                
                try:
                    with torch.no_grad():
                        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)
                        
                        generated = model.base_model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=100,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        response = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        
                        # Simple answer extraction
                        pred_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', response)
                        true_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
                        
                        if pred_nums and true_nums:
                            pred = pred_nums[-1].replace(',', '')
                            true = true_nums[-1].replace(',', '')
                            
                            try:
                                if abs(float(pred) - float(true)) < 0.01:
                                    val_correct += 1
                            except:
                                pass
                except:
                    continue
            
            model.train()
            
            val_accuracy = val_correct / val_total
            print(f"📊 Validation Accuracy: {val_accuracy:.2%} ({val_correct}/{val_total})")
            
            metrics['val_accuracy'].append(val_accuracy)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"🏆 New best validation accuracy!")
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'metrics': metrics,
                    'specialization_patterns': dict(specialization_patterns)
                }, os.path.join(checkpoint_dir, 'best_generalized_coconut.pt'))
        
        scheduler.step()
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.2%}")
    
    # Training statistics
    print(f"\n📈 Training Statistics:")
    print(f"   • Total samples processed: {total_successful_samples}")
    print(f"   • Total samples failed: {total_failed_samples}")
    print(f"   • Overall success rate: {total_successful_samples/max(1, total_successful_samples + total_failed_samples)*100:.1f}%")
    
    if total_successful_samples < (total_successful_samples + total_failed_samples) * 0.5:
        print("\n⚠️ WARNING: Less than 50% of samples were successfully processed!")
        print("   This indicates issues with the training pipeline.")
        print("   Recommendations:")
        print(f"   • Reduce batch_size (currently {batch_size})")
        print(f"   • Reduce max_trajectories (currently {MAX_TRAJECTORIES})")
        print("   • Check GPU memory availability")
    
    # Save final metrics
    metrics['total_successful'] = total_successful_samples
    metrics['total_failed'] = total_failed_samples
    
    with open(os.path.join(checkpoint_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📁 Checkpoints saved to: {checkpoint_dir}")
    print("="*70)
    
    return model, metrics


# ============================================================
# DEBUGGING AND TESTING FUNCTIONS
# ============================================================

def test_single_sample(model, tokenizer, dataset, device):
    """Test a single sample to debug issues"""
    print("\n" + "="*70)
    print("SINGLE SAMPLE DEBUG TEST")
    print("="*70)
    
    model.train()
    
    # Get one sample
    item = dataset[0]
    question = item['question']
    answer_text = item['answer']
    
    print(f"\n📝 Sample:")
    print(f"   Question: {question[:100]}...")
    print(f"   Answer: {answer_text[:100]}...")
    
    train_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution: {answer_text}"
    
    try:
        # Tokenize
        print("\n1️⃣ Tokenizing...")
        inputs = tokenizer(
            train_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=False
        ).to(device)
        print(f"   ✅ Input shape: {inputs['input_ids'].shape}")
        
        # Forward pass
        print("\n2️⃣ Forward pass...")
        with torch.autograd.set_detect_anomaly(True):
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids'],
                temperature=1.0,
                return_trajectory_info=True
            )
        
        print(f"   ✅ Got outputs")
        
        # Check outputs
        print("\n3️⃣ Output analysis:")
        if outputs is None:
            print("   ❌ Outputs is None!")
            return False
        
        if 'error' in outputs:
            print(f"   ❌ Error in outputs: {outputs['error']}")
            return False
        
        if 'loss' not in outputs:
            print("   ❌ No loss in outputs!")
            print(f"   Available keys: {outputs.keys()}")
            return False
        
        loss = outputs['loss']
        print(f"   • Loss: {loss.item() if loss is not None else 'None'}")
        print(f"   • Loss requires grad: {loss.requires_grad if loss is not None else 'N/A'}")
        print(f"   • Trajectories: {outputs.get('avg_trajectories', 'N/A')}")
        print(f"   • Complexity: {outputs.get('avg_complexity', 'N/A')}")
        print(f"   • Diversity: {outputs.get('avg_diversity', 'N/A')}")
        print(f"   • Trajectory length: {outputs.get('trajectory_length', 'N/A')}")
        
        # Try backward
        print("\n4️⃣ Backward pass...")
        if loss is not None and loss.requires_grad:
            loss.backward()
            print("   ✅ Backward pass successful")
            
            # Check if gradients were computed
            has_grads = False
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    has_grads = True
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        print(f"   ✅ Found non-zero gradient in: {name[:50]}... (norm: {grad_norm:.6f})")
                        break
            
            if not has_grads:
                print("   ⚠️ No gradients found!")
        else:
            print("   ❌ Loss doesn't require gradient or is None")
            return False
        
        print("\n✅ Single sample test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Generalized Multi-Trajectory COCONUT Implementation")
    print("Trajectories naturally discover their own specializations")
    print("No pre-specified reasoning strategies - pure emergent behavior")
    
    # Parse command line arguments
    debug = '--debug' in sys.argv or '-d' in sys.argv
    test_only = '--test' in sys.argv or '-t' in sys.argv
    
    if test_only:
        # Just run single sample test
        print("\n🔬 Running single sample test...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
        tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        print("Loading model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B-Instruct',
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_cache=False
        )
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Create COCONUT model
        print("Creating COCONUT model...")
        model = GeneralizedMultiTrajectoryCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
            start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
            end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
            eos_token_id=tokenizer.eos_token_id,
            hidden_size=4096,
            reasoning_dim=256,
            min_trajectories=2,
            max_trajectories=3,  # Reduced for testing
            max_latent_steps=3,  # Reduced for testing
            dropout_rate=0.1
        )
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("gsm8k", "main", split="train")
        
        # Run test
        success = test_single_sample(model, tokenizer, dataset, device)
        
        if success:
            print("\n✅ Test successful! The model can process samples.")
        else:
            print("\n❌ Test failed! There are issues with the model.")
        
        sys.exit(0 if success else 1)
    
    # Normal training
    if debug:
        print("\n🔍 Running in DEBUG mode with minimal samples")
    
    model, metrics = train_generalized_coconut(debug_mode=debug)
