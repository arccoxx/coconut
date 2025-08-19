#!/usr/bin/env python3
"""
Generalized Multi-Trajectory COCONUT - Fully Debugged & Patched
==================================================================
Fixed critical bugs:
1. Stop votes padding now uses torch.tensor([0.0]) for shape [1]
2. Ensured all tensors in stacking have consistent shapes [1]
3. Added unsqueeze where needed for scalars
4. Completed validation loop with full generation and answer extraction
5. Improved dimension safety in squeezes and stacks
6. Fixed thought dimension consistency (project if needed)
7. Better error handling with raises instead of silent fails where appropriate

author: aidan grok claude
"""
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
warnings.filterwarnings("ignore", category=UserWarning)
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
   
    discovered_strategy: Optional[str] = None
    exploration_temperature: float = 1.0
    risk_tolerance: float = 0.5
   
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
    def ensure_batch_dim(tensor: torch.Tensor, batch_size: int = 1, device=None, dtype=None) -> torch.Tensor:
        """Ensure tensor has proper batch dimension"""
        if tensor is None:
            return None
       
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
       
        current_batch = tensor.size(0)
        if current_batch != batch_size:
            if current_batch == 1:
                tensor = tensor.expand(batch_size, *tensor.shape[1:])
            elif batch_size == 1:
                tensor = tensor[:1]
            elif current_batch > batch_size:
                tensor = tensor[:batch_size]
            else:
                repeat_times = [batch_size // current_batch] + [1] * (tensor.dim() - 1)
                tensor = tensor.repeat(*repeat_times)
                if tensor.size(0) > batch_size:
                    tensor = tensor[:batch_size]
       
        if device is not None:
            tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
   
    @staticmethod
    def safe_squeeze(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Safely squeeze tensor only if dimension size is 1"""
        if tensor is None:
            return None
        if dim is None:
            return tensor.squeeze()
        else:
            if tensor.dim() > dim and tensor.size(dim) == 1:
                return tensor.squeeze(dim)
        return tensor
   
    @staticmethod
    def safe_cat(tensors: List[torch.Tensor], dim: int = -1, device=None, dtype=None) -> torch.Tensor:
        """Safely concatenate tensors with dimension checking"""
        if not tensors:
            return None
       
        tensors = [t for t in tensors if t is not None]
        if not tensors:
            return None
       
        if device is not None or dtype is not None:
            tensors = [
                t.to(device=device if device else t.device,
                     dtype=dtype if dtype else t.dtype)
                for t in tensors
            ]
       
        max_dim = max(t.dim() for t in tensors)
        aligned = []
        for t in tensors:
            while t.dim() < max_dim:
                t = t.unsqueeze(0)
            aligned.append(t)
       
        return torch.cat(aligned, dim=dim)
   
    @staticmethod
    def safe_stack(tensors: List[torch.Tensor], dim: int = 0, device=None, dtype=None) -> torch.Tensor:
        """Safely stack tensors with dimension checking"""
        if not tensors:
            return None
       
        if device is not None or dtype is not None:
            tensors = [
                t.to(device=device if device else t.device,
                     dtype=dtype if dtype else t.dtype)
                for t in tensors
            ]
       
        # Ensure all have same dim
        max_dim = max(t.dim() for t in tensors)
        tensors = [t.view(*([1] * (max_dim - t.dim())), *t.shape) if t.dim() < max_dim else t for t in tensors]
       
        # Find reference shape, ignoring the stack dim
        ref_shape = list(tensors[0].shape)
        ref_shape.pop(dim if dim >= 0 else len(ref_shape) + dim)
       
        for i in range(len(tensors)):
            current_shape = list(tensors[i].shape)
            current_shape.pop(dim if dim >= 0 else len(current_shape) + dim)
            if current_shape != ref_shape:
                # Attempt to broadcast
                new_shape = list(tensors[i].shape)
                for j in range(len(ref_shape)):
                    if current_shape[j] != ref_shape[j] and current_shape[j] == 1:
                        new_shape.insert(j, ref_shape[j])
                tensors[i] = tensors[i].expand(new_shape)
       
        return torch.stack(tensors, dim=dim)
# ============================================================
# TRAJECTORY SPECIALIZATION SYSTEM
# ============================================================
class TrajectorySpecializationTracker:
    """Tracks and encourages natural specialization of trajectories"""
   
    def __init__(self, num_trajectories: int, feature_dim: int, device=None, dtype=None):
        self.num_trajectories = num_trajectories
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.bfloat16
       
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
       
        sig['avg_step_size'].append(step_size)
        sig['confidence_profile'].append(confidence)
       
        if len(sig['exploration_pattern']) > 0:
            last_dir = sig['exploration_pattern'][-1]
            if isinstance(last_dir, torch.Tensor):
                last_dir = last_dir.to(direction.device, direction.dtype)
                dir_change = 1.0 - F.cosine_similarity(
                    direction.view(1, -1),
                    last_dir.view(1, -1)
                ).item()
                sig['direction_variance'].append(dir_change)
       
        sig['exploration_pattern'].append(direction.detach().to(self.device, self.dtype))
       
        if value > 0.5:
            sig['success_contexts'].append(context.detach().to(self.device, self.dtype))
        else:
            sig['failure_contexts'].append(context.detach().to(self.device, self.dtype))
   
    def compute_specialization_bonus(self, traj_idx: int) -> float:
        """Compute bonus for trajectory based on specialization"""
        sig = self.trajectory_signatures[traj_idx]
       
        if len(sig['avg_step_size']) < 10:
            return 0.0
       
        avg_step = np.mean(list(sig['avg_step_size']))
        step_consistency = 1.0 / (np.std(list(sig['avg_step_size'])) + 1e-6)
        avg_confidence = np.mean(list(sig['confidence_profile']))
       
        uniqueness = 0.0
        for other_idx in range(self.num_trajectories):
            if other_idx == traj_idx:
                continue
           
            other_sig = self.trajectory_signatures[other_idx]
            if len(other_sig['avg_step_size']) < 10:
                continue
           
            other_avg_step = np.mean(list(other_sig['avg_step_size']))
            other_confidence = np.mean(list(other_sig['confidence_profile']))
           
            diff = abs(avg_step - other_avg_step) + abs(avg_confidence - other_confidence)
            uniqueness += diff
       
        return min(uniqueness / max(1, self.num_trajectories - 1), 1.0) * 0.2
   
    def get_diversity_matrix(self) -> torch.Tensor:
        """Get matrix of pairwise trajectory diversity"""
        diversity = torch.zeros(self.num_trajectories, self.num_trajectories,
                              device=self.device, dtype=self.dtype)
       
        for i in range(self.num_trajectories):
            for j in range(i + 1, self.num_trajectories):
                sig_i = self.trajectory_signatures[i]
                sig_j = self.trajectory_signatures[j]
               
                if len(sig_i['avg_step_size']) > 0 and len(sig_j['avg_step_size']) > 0:
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
    """Graph-based memory for general reasoning patterns"""
   
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
        self.device = device if device is not None else torch.device('cpu')
       
        self.nodes = torch.zeros(memory_size, reasoning_dim, dtype=self.dtype, device=self.device)
        self.node_values = torch.full((memory_size,), -float('inf'), dtype=self.dtype, device=self.device)
        self.node_timestamps = torch.zeros(memory_size, dtype=torch.long, device=self.device)
        self.adjacency_matrix = torch.zeros(memory_size, memory_size, dtype=self.dtype, device=self.device)
        self.node_creators = torch.zeros(memory_size, dtype=torch.long, device=self.device)
       
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=reasoning_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
            dtype=self.dtype,
            device=self.device
        )
       
        self.relation_encoder = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim // 2, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1, dtype=self.dtype, device=self.device),
            nn.Sigmoid()
        )
       
        self.memory_evolution = nn.GRUCell(reasoning_dim, reasoning_dim, dtype=self.dtype, device=self.device)
       
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, reasoning_dim // 4, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 4, 32, dtype=self.dtype, device=self.device)
        )
       
        self.memory_ptr = 0
        self.total_writes = 0
        self.memory_filled = False
   
    def write(
        self,
        position: torch.Tensor,
        value: torch.Tensor,
        source_trajectory: int,
        related_positions: Optional[List[torch.Tensor]] = None
    ) -> int:
        """Write to memory and establish graph relationships"""
        position = DimensionHelper.ensure_batch_dim(position, 1, self.device, self.dtype)
        position = DimensionHelper.safe_squeeze(position, 0)
       
        if position.size(-1) != self.reasoning_dim:
            raise ValueError(f"Position dimension {position.size(-1)} doesn't match reasoning_dim {self.reasoning_dim}")
       
        with torch.no_grad():
            self.nodes[self.memory_ptr] = position
           
            if value.dim() > 0:
                value = value.view(-1)[0]
            self.node_values[self.memory_ptr] = value
            self.node_timestamps[self.memory_ptr] = self.total_writes
            self.node_creators[self.memory_ptr] = source_trajectory
           
            current_idx = self.memory_ptr
           
            if related_positions and len(related_positions) > 0:
                for related_pos in related_positions:
                    if related_pos is not None:
                        related_pos = DimensionHelper.ensure_batch_dim(related_pos, 1, self.device, self.dtype)
                        related_pos = DimensionHelper.safe_squeeze(related_pos, 0)
                       
                        if related_pos.size(-1) != self.reasoning_dim:
                            continue
                       
                        similarities = F.cosine_similarity(
                            related_pos.unsqueeze(0),
                            self.nodes[:self.memory_ptr if not self.memory_filled else self.memory_size],
                            dim=1
                        )
                       
                        k = min(3, self.memory_ptr if not self.memory_filled else self.memory_size)
                        if k > 0:
                            top_k = torch.topk(similarities, k)
                            for idx in top_k.indices:
                                if idx.item() != current_idx:
                                    combined = torch.cat([
                                        self.nodes[current_idx],
                                        self.nodes[idx.item()]
                                    ])
                                    strength = self.relation_encoder(combined).item()
                                   
                                    self.adjacency_matrix[current_idx, idx.item()] = strength
                                    self.adjacency_matrix[idx.item(), current_idx] = strength
           
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
            if self.memory_ptr == 0:
                self.memory_filled = True
           
            self.total_writes += 1
           
            return current_idx
   
    def retrieve_graph_context(
        self,
        query: torch.Tensor,
        num_hops: int = 2,
        top_k: int = 5,
        source_trajectory: Optional[int] = None,
        debug_mode: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Retrieve context using graph traversal"""
        query = DimensionHelper.ensure_batch_dim(query, 1, self.device, self.dtype)
       
        if query.size(-1) != self.reasoning_dim:
            raise ValueError(f"Query dimension {query.size(-1)} doesn't match reasoning_dim {self.reasoning_dim}")
       
        if not self.memory_filled and self.memory_ptr < top_k:
            return torch.zeros(self.reasoning_dim, device=self.device, dtype=self.dtype), {}
       
        valid_nodes = self.memory_ptr if not self.memory_filled else self.memory_size
       
        query_norm = F.normalize(query, p=2, dim=-1)
        nodes_norm = F.normalize(self.nodes[:valid_nodes], p=2, dim=-1)
        similarities = torch.matmul(query_norm, nodes_norm.t()).squeeze(0)
       
        recency_weight = torch.exp(-0.01 * (self.total_writes - self.node_timestamps[:valid_nodes]).float()).to(self.device, self.dtype)
        value_weight = torch.sigmoid(self.node_values[:valid_nodes]).to(self.device, self.dtype)
       
        diversity_bonus = torch.ones(valid_nodes, device=self.device, dtype=self.dtype)
        if source_trajectory is not None:
            different_creator = (self.node_creators[:valid_nodes] != source_trajectory).float().to(self.device, self.dtype)
            diversity_bonus += different_creator * 0.2
       
        weighted_sim = similarities * value_weight * recency_weight * diversity_bonus
       
        k = min(top_k, valid_nodes)
        top_k_scores, top_k_indices = torch.topk(weighted_sim, k)
       
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
       
        if context_nodes:
            context_indices = torch.tensor(context_nodes[:min(20, len(context_nodes))],
                                          device=self.device, dtype=torch.long)
            context_memories = self.nodes[context_indices]
           
            query_for_attn = query.unsqueeze(1) if query.dim() == 2 else query.unsqueeze(0).unsqueeze(1)
            context_for_attn = context_memories.unsqueeze(0)
           
            if query_for_attn.size(0) != context_for_attn.size(0):
                context_for_attn = context_for_attn.expand(query_for_attn.size(0), -1, -1)
           
            attended, attention_weights = self.graph_attention(
                query_for_attn,
                context_for_attn,
                context_for_attn
            )
           
            attended_2d = attended.squeeze(1)
           
            evolved = self.memory_evolution(
                DimensionHelper.safe_squeeze(query),
                DimensionHelper.safe_squeeze(attended_2d)
            )
           
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
       
        return torch.zeros(self.reasoning_dim, device=self.device, dtype=self.dtype), {}
   
    def get_trajectory_contribution_stats(self) -> Dict[int, float]:
        """Get statistics about trajectory contributions"""
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
        device=None,
        dtype=None
    ):
        self.min_trajectories = min_trajectories
        self.max_trajectories = max_trajectories
        self.prune_threshold = prune_threshold
        self.spawn_threshold = spawn_threshold
        self.diversity_bonus = diversity_bonus
        self.device = device
        self.dtype = dtype
       
        self.trajectory_history = defaultdict(list)
        self.trajectory_lineage = {}
        self.specialization_tracker = TrajectorySpecializationTracker(
            max_trajectories, 256, device=device, dtype=dtype
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
        std_value = np.std(values) if len(values) > 1 else 0.1
       
        diversity_matrix = self.specialization_tracker.get_diversity_matrix()
       
        for i, traj in enumerate(trajectories):
            if not traj.is_active:
                continue
           
            spec_bonus = self.specialization_tracker.compute_specialization_bonus(i)
            adjusted_value = traj.value + spec_bonus
           
            if len(trajectories) - len(to_prune) > self.min_trajectories:
                if i < diversity_matrix.size(0):
                    avg_diversity = diversity_matrix[i].mean().item()
                else:
                    avg_diversity = 0.5
               
                if adjusted_value < mean_value - std_value * self.prune_threshold and avg_diversity < 0.5:
                    to_prune.append(i)
                    continue
           
            if len(trajectories) + len(to_spawn) - len(to_prune) < self.max_trajectories:
                if adjusted_value > mean_value + std_value * self.spawn_threshold:
                    to_spawn.append(i)
       
        while len(trajectories) - len(to_prune) < self.min_trajectories and to_prune:
            to_prune.pop()
       
        return to_prune, to_spawn
   
    def spawn_trajectory(
        self,
        parent: TrajectoryState,
        mutation_strength: float = 0.1
    ) -> TrajectoryState:
        """Spawn new trajectory with inherited characteristics"""
        new_temperature = parent.exploration_temperature * (1 + np.random.randn() * 0.1)
        new_temperature = np.clip(new_temperature, 0.5, 2.0)
       
        new_risk = parent.risk_tolerance * (1 + np.random.randn() * 0.1)
        new_risk = np.clip(new_risk, 0.1, 0.9)
       
        noise = torch.randn_like(parent.position, device=parent.position.device,
                                dtype=parent.position.dtype) * mutation_strength * (1 + new_risk)
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
   
    def compute_diversity(self, positions: List[torch.Tensor]) -> float:
        """Compute diversity of trajectory positions"""
        if not positions or len(positions) <= 1:
            return 1.0
       
        positions_stacked = DimensionHelper.safe_stack(positions, device=self.device, dtype=self.dtype)
        if positions_stacked.dim() == 1:
            positions_stacked = positions_stacked.unsqueeze(0)
       
        distances = torch.cdist(positions_stacked, positions_stacked)
       
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        if mask.sum() > 0:
            mean_distance = distances[mask].mean().item()
        else:
            mean_distance = 0.0
       
        diversity = torch.sigmoid(torch.tensor(mean_distance * 2, device=self.device)).item()
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
        dtype=None,
        device=None
    ):
        super().__init__()
        self.min_trajectories = min_trajectories
        self.max_trajectories = max_trajectories
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.device = device
       
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1, dtype=self.dtype, device=self.device),
            nn.Sigmoid()
        )
       
        self.traj_count_net = nn.Sequential(
            nn.Linear(hidden_size + 1, 64, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(64, max_trajectories - min_trajectories + 1, dtype=self.dtype, device=self.device),
            nn.Softmax(dim=-1)
        )
       
        self.complexity_history = deque(maxlen=100)
   
    def forward(
        self,
        initial_state: torch.Tensor,
        return_all: bool = False
    ) -> Union[Tuple[int, float], Tuple[torch.Tensor, torch.Tensor]]:
        """Analyze complexity with proper batch handling"""
        initial_state = DimensionHelper.ensure_batch_dim(initial_state, device=self.device, dtype=self.dtype)
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
            num_traj = trajectory_counts[0].item() if trajectory_counts.dim() > 0 else int(trajectory_counts)
            complexity = complexity_scores[0].item() if complexity_scores.dim() > 0 else float(complexity_scores)
            return num_traj, complexity
# ============================================================
# GENERALIZED MULTI-TRAJECTORY NAVIGATOR
# ============================================================
class GeneralizedMultiTrajectoryNavigator(nn.Module):
    """Navigator for general reasoning with trajectory specialization"""
   
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        min_trajectories: int = 2,
        max_trajectories: int = 5,
        memory_size_per_trajectory: int = 50,
        shared_memory_size: int = 200,
        dropout_rate: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.min_trajectories = min_trajectories
        self.max_trajectories = max_trajectories
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.bfloat16
       
        # Core projection layers
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, reasoning_dim, dtype=self.dtype, device=self.device)
        )
       
        self.thought_projection = nn.Sequential(
            nn.Linear(reasoning_dim, hidden_size // 4, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size, dtype=self.dtype, device=self.device)
        )
       
        # Trajectory initialization diversity network
        self.trajectory_initializers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(reasoning_dim, reasoning_dim, dtype=self.dtype, device=self.device),
                nn.Tanh(),
                nn.Linear(reasoning_dim, reasoning_dim, dtype=self.dtype, device=self.device)
            ) if i % 2 == 0 else
            nn.Sequential(
                nn.Linear(reasoning_dim, reasoning_dim * 2, dtype=self.dtype, device=self.device),
                nn.ReLU(),
                nn.Linear(reasoning_dim * 2, reasoning_dim, dtype=self.dtype, device=self.device)
            )
            for i in range(max_trajectories)
        ])
       
        # Graph-based shared memory
        self.graph_memory = GraphMemoryBank(
            memory_size=shared_memory_size,
            reasoning_dim=reasoning_dim,
            num_heads=4,
            device=self.device,
            dtype=self.dtype
        )
       
        # Enhanced complexity analyzer
        self.complexity_analyzer = EnhancedComplexityAnalyzer(
            hidden_size, min_trajectories, max_trajectories, dtype=self.dtype, device=self.device
        )
       
        # Lifecycle manager with specialization
        self.lifecycle_manager = TrajectoryLifecycleManager(
            min_trajectories, max_trajectories, device=self.device, dtype=self.dtype
        )
       
        # Trajectory-specific heads with learnable specialization
        self.trajectory_heads = nn.ModuleList([
            nn.ModuleDict({
                'continue': nn.Sequential(
                    nn.Linear(reasoning_dim * 3, reasoning_dim, dtype=self.dtype, device=self.device),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(reasoning_dim, 2, dtype=self.dtype, device=self.device)
                ),
                'direction': nn.Linear(reasoning_dim * 3, reasoning_dim, dtype=self.dtype, device=self.device),
                'step_size': nn.Linear(reasoning_dim * 3, 1, dtype=self.dtype, device=self.device),
                'value': nn.Linear(reasoning_dim * 3, 1, dtype=self.dtype, device=self.device),
                'confidence': nn.Linear(reasoning_dim * 3, 1, dtype=self.dtype, device=self.device),
                'specialization': nn.Sequential(
                    nn.Linear(reasoning_dim * 3, reasoning_dim // 2, dtype=self.dtype, device=self.device),
                    nn.ReLU(),
                    nn.Linear(reasoning_dim // 2, 16, dtype=self.dtype, device=self.device),
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
            dtype=self.dtype,
            device=self.device
        )
       
        # Diversity reward network
        self.diversity_evaluator = nn.Sequential(
            nn.Linear(reasoning_dim * max_trajectories, reasoning_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim, 1, dtype=self.dtype, device=self.device),
            nn.Sigmoid()
        )
       
        # Graph-inspired trajectory interaction
        self.trajectory_graph_encoder = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim, reasoning_dim // 2, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1, dtype=self.dtype, device=self.device),
            nn.Sigmoid()
        )
       
        # Ensemble aggregation with specialization awareness
        self.ensemble_gate = nn.Sequential(
            nn.Linear(reasoning_dim * (max_trajectories + 2), reasoning_dim, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Linear(reasoning_dim, max_trajectories + 1, dtype=self.dtype, device=self.device),
            nn.Softmax(dim=-1)
        )
       
        self.dropout = nn.Dropout(dropout_rate)
       
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
        state = DimensionHelper.ensure_batch_dim(state, 1, self.device, self.dtype)
       
        # Step 1: Analyze complexity
        num_trajectories, complexity_score = self.complexity_analyzer(state, return_all=False)
       
        # Step 2: Retrieve graph context
        graph_context = None
        graph_metadata = {}
       
        if self.graph_memory.total_writes > 0:
            state_projected = self.state_projection(state)
            graph_context, graph_metadata = self.graph_memory.retrieve_graph_context(
                state_projected, num_hops=2, top_k=5, source_trajectory=0
            )
       
        # Step 3: Initialize trajectories with diverse strategies
        trajectory_states = []
        active_trajectories = []
       
        for traj_idx in range(num_trajectories):
            traj_state = self.state_projection(state)
           
            initializer = self.trajectory_initializers[traj_idx % len(self.trajectory_initializers)]
            traj_state = initializer(traj_state)
           
            if graph_context is not None:
                graph_context = DimensionHelper.ensure_batch_dim(graph_context, 1, self.device, self.dtype)
                weight = 0.1 + (traj_idx / num_trajectories) * 0.3
                traj_state = (1 - weight) * traj_state + weight * graph_context
           
            exploration_temp = 0.5 + (traj_idx / max(1, num_trajectories - 1)) * 1.5
            risk_tolerance = 0.2 + (traj_idx / max(1, num_trajectories - 1)) * 0.6
           
            if self.training or traj_idx > 0:
                noise_scale = 0.1 * risk_tolerance * (1 - step_num / 10)
                noise = torch.randn_like(traj_state, device=self.device, dtype=self.dtype) * noise_scale
                traj_state = traj_state + noise
           
            trajectory_states.append(traj_state)
           
            traj_obj = TrajectoryState(
                position=traj_state,
                value=0.5,
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
           
            stacked_states = DimensionHelper.safe_stack(trajectory_states, dim=1, device=self.device, dtype=self.dtype)
           
            attended_states = []
            for i in range(num_trajectories):
                query = stacked_states[:, i:i+1, :]
               
                key_value_weights = trajectory_graph[i].unsqueeze(0).unsqueeze(-1).to(self.device, self.dtype)
                weighted_kv = stacked_states * key_value_weights
               
                attended, attn_weights = self.cross_attention(query, weighted_kv, weighted_kv)
                attended_states.append(attended.squeeze(1))
           
            trajectory_states = attended_states
       
        # Step 5: Lifecycle management
        if step_num > 2 and step_num % 2 == 0:
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
                   
                    child_state = trajectory_states[idx] + torch.randn_like(trajectory_states[idx],
                                                                           device=self.device,
                                                                           dtype=self.dtype) * 0.1
                    trajectory_states.append(child_state)
       
        # Step 6: Navigate each active trajectory
        trajectory_outputs = []
        related_positions = []
       
        for traj_idx, (traj_state, traj_obj) in enumerate(zip(trajectory_states, active_trajectories)):
            if not traj_obj.is_active:
                continue
           
            traj_state = DimensionHelper.ensure_batch_dim(traj_state, 1, self.device, self.dtype)
           
            heads = self.trajectory_heads[traj_idx % self.max_trajectories]
           
            if graph_context is not None:
                graph_context = DimensionHelper.ensure_batch_dim(graph_context, 1, self.device, self.dtype)
                decision_input = torch.cat([traj_state, traj_state, graph_context], dim=-1)
            else:
                padding = torch.zeros_like(traj_state, device=self.device, dtype=self.dtype)
                decision_input = torch.cat([traj_state, traj_state, padding], dim=-1)
           
            traj_temp = temperature * traj_obj.exploration_temperature
           
            continue_logits = heads['continue'](decision_input) / traj_temp
            continue_probs = F.softmax(continue_logits, dim=-1)
           
            if self.training:
                continue_dist = torch.distributions.Categorical(continue_probs)
                continue_action = continue_dist.sample()
            else:
                continue_action = continue_probs.argmax(dim=-1)
           
            direction = F.normalize(heads['direction'](decision_input), p=2, dim=-1)
            step_size = torch.sigmoid(heads['step_size'](decision_input)) * 2.0 * traj_obj.risk_tolerance
            value = heads['value'](decision_input)
            confidence = torch.sigmoid(heads['confidence'](decision_input))
           
            specialization = heads['specialization'](decision_input)
           
            next_position = traj_state + step_size * direction
            latent_thought = self.thought_projection(next_position)
           
            step_size_scalar = step_size.item()
            value_scalar = value.item()
            confidence_scalar = confidence.item()
           
            self.lifecycle_manager.specialization_tracker.update_trajectory_signature(
                traj_idx,
                step_size_scalar,
                DimensionHelper.safe_squeeze(direction),
                value_scalar,
                confidence_scalar,
                DimensionHelper.safe_squeeze(traj_state)
            )
           
            traj_obj.position = next_position
            traj_obj.value = value_scalar
            traj_obj.confidence = confidence_scalar
            traj_obj.step_count = step_num
           
            related_positions.append(next_position)
           
            trajectory_outputs.append({
                'thought': latent_thought,
                'position': next_position,
                'stop_vote': continue_action == 1,
                'continue_prob': continue_probs[:, 0].item() if continue_probs.dim() > 1 else continue_probs.item(),
                'value': value,
                'confidence': confidence,
                'trajectory_idx': traj_idx,
                'trajectory_obj': traj_obj,
                'specialization': specialization
            })
       
        # Step 7: Compute diversity score
        if trajectory_outputs and len(trajectory_outputs) > 1:
            all_positions = []
            for out in trajectory_outputs:
                pos = out['position']
                pos = DimensionHelper.safe_squeeze(pos)
                if pos.dim() == 0:
                    pos = pos.unsqueeze(0)
                all_positions.append(pos)
           
            while len(all_positions) < self.max_trajectories:
                all_positions.append(torch.zeros_like(all_positions[0], device=self.device, dtype=self.dtype))
           
            concat_positions = torch.cat(all_positions[:self.max_trajectories])
            diversity_score = self.diversity_evaluator(concat_positions.unsqueeze(0)).item()
        else:
            diversity_score = 0.0
       
        # Step 8: Write best trajectories to graph memory
        if trajectory_outputs:
            values = [out['value'].view(-1)[0].item() for out in trajectory_outputs]
           
            adjusted_values = []
            for i, v in enumerate(values):
                spec_bonus = self.lifecycle_manager.specialization_tracker.compute_specialization_bonus(i)
                adjusted_values.append(v + spec_bonus + diversity_score * 0.1)
           
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
   
    def _build_trajectory_graph(self, trajectory_states: List[torch.Tensor]) -> torch.Tensor:
        """Build graph relationships between trajectories"""
        num_traj = len(trajectory_states)
       
        states = [DimensionHelper.ensure_batch_dim(s, 1, self.device, self.dtype) for s in trajectory_states]
        states = [DimensionHelper.safe_squeeze(s) for s in states]
       
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
                torch.tensor([True], device=self.device, dtype=torch.bool)
            )
       
        # Aggregate thoughts
        thoughts = []
        for out in trajectory_outputs:
            thought = out['thought']
            thought = DimensionHelper.ensure_batch_dim(thought, 1, self.device, self.dtype)
            thought = DimensionHelper.safe_squeeze(thought, 0)
           
            if thought.size(-1) != self.hidden_size:
                if thought.size(-1) == self.reasoning_dim:
                    thought = self.thought_projection(thought.unsqueeze(0)).squeeze(0)
            thoughts.append(thought)
       
        while len(thoughts) < self.max_trajectories:
            thoughts.append(torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype))
       
        thoughts_tensor = DimensionHelper.safe_stack(thoughts[:self.max_trajectories], device=self.device, dtype=self.dtype)
       
        all_states = [out['position'] for out in trajectory_outputs]
        all_states = [DimensionHelper.ensure_batch_dim(s, 1, self.device, self.dtype) for s in all_states]
        all_states = [DimensionHelper.safe_squeeze(s) for s in all_states]
       
        while len(all_states) < self.max_trajectories:
            all_states.append(torch.zeros(self.reasoning_dim, device=self.device, dtype=self.dtype))
       
        concat_states = torch.cat(all_states[:self.max_trajectories])
       
        if graph_context is not None:
            graph_context = DimensionHelper.ensure_batch_dim(graph_context, 1, self.device, self.dtype)
            graph_context = DimensionHelper.safe_squeeze(graph_context)
            ensemble_input = torch.cat([concat_states, graph_context, graph_context])
        else:
            padding = torch.zeros(self.reasoning_dim * 2, device=self.device, dtype=self.dtype)
            ensemble_input = torch.cat([concat_states, padding])
       
        ensemble_weights = self.ensemble_gate(ensemble_input.unsqueeze(0))
       
        if thoughts_tensor.dim() == 2:
            thoughts_tensor = thoughts_tensor.unsqueeze(0)
       
        weighted_thoughts = thoughts_tensor * ensemble_weights[:, :self.max_trajectories].unsqueeze(-1)
        ensemble_thought = weighted_thoughts.sum(dim=1).squeeze(0)
       
        # FIXED: Stop votes handling with consistent shapes
        stop_votes = [out['stop_vote'].float() for out in trajectory_outputs]
        stop_votes = [v.unsqueeze(0) if v.dim() == 0 else v for v in stop_votes]
       
        while len(stop_votes) < self.max_trajectories:
            stop_votes.append(torch.tensor([0.0], device=self.device, dtype=self.dtype))
       
        stop_tensor = torch.stack(stop_votes[:self.max_trajectories])
        weighted_stop = (stop_tensor * ensemble_weights[0, :self.max_trajectories]).sum() > 0.5
       
        return ensemble_thought, torch.tensor([weighted_stop], device=self.device, dtype=torch.bool)
# ============================================================
# COMPLETE MODEL
# ============================================================
class GeneralizedMultiTrajectoryCoconut(nn.Module):
    """Complete model for general reasoning with trajectory specialization"""
   
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
        force_reasoning: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with trajectory specialization"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
       
        dtype = next(self.parameters()).dtype
       
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
            initial_states = last_hidden.mean(dim=1).to(dtype)
       
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
                'stop_patterns': [],
                'actual_steps': 0
            }
           
            ensemble_sequence = []
            actual_steps = 0
           
            min_steps = 2 if force_reasoning else 1
            for step in range(self.max_latent_steps):
                nav_output = self.navigator.navigate_advanced(
                    current_state,
                    step_num=step,
                    temperature=temperature,
                    batch_idx=b
                )
               
                actual_steps += 1
               
                trajectory_info['num_trajectories'].append(nav_output.get('num_trajectories', 0))
                trajectory_info['num_active'].append(nav_output.get('num_active', 0))
                trajectory_info['complexity_scores'].append(nav_output.get('complexity_score', 0))
                trajectory_info['diversity_scores'].append(nav_output.get('diversity_score', 0))
                trajectory_info['specialization_info'].append(nav_output.get('specialization_info', {}))
               
                stop_value = nav_output['ensemble_stop']
                should_stop = stop_value.item() if hasattr(stop_value, 'item') else bool(stop_value)
               
                trajectory_info['stop_patterns'].append(should_stop)
               
                if step < min_steps - 1:
                    should_stop = False
               
                if should_stop or step == self.max_latent_steps - 1:
                    break
               
                ensemble_thought = nav_output['ensemble_thought']
                ensemble_thought = DimensionHelper.ensure_batch_dim(ensemble_thought, 1, device, dtype)
                ensemble_thought = DimensionHelper.safe_squeeze(ensemble_thought, 0)
               
                ensemble_sequence.append(ensemble_thought)
                trajectory_info['ensemble_thoughts'].append(ensemble_thought)
               
                current_state = ensemble_thought
           
            trajectory_info['actual_steps'] = actual_steps
           
            batch_trajectory_info.append(trajectory_info)
            batch_ensemble_sequences.append(ensemble_sequence)
       
        max_latent_len = max(len(seq) for seq in batch_ensemble_sequences) if batch_ensemble_sequences else min_steps
       
        if max_latent_len < min_steps and force_reasoning:
            for b in range(batch_size):
                while len(batch_ensemble_sequences[b]) < min_steps:
                    dummy_thought = torch.zeros(self.hidden_size, device=device, dtype=dtype)
                    batch_ensemble_sequences[b].append(dummy_thought)
            max_latent_len = min_steps
       
        if max_latent_len > 0:
            if labels is not None:
                batch_size, seq_len = labels.shape
                new_labels = torch.full(
                    (batch_size, seq_len + max_latent_len),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )
                new_labels[:, :seq_len] = labels
                labels = new_labels
           
            padded_sequences = []
            for seq in batch_ensemble_sequences:
                seq = [DimensionHelper.ensure_batch_dim(s, 1, device, dtype) for s in seq]
                seq = [DimensionHelper.safe_squeeze(s, 0) for s in seq]
               
                if len(seq) < max_latent_len:
                    padding = [torch.zeros_like(seq[0], device=device, dtype=dtype) for _ in range(max_latent_len - len(seq))]
                    padded_seq = seq + padding
                else:
                    padded_seq = seq
               
                stacked_seq = DimensionHelper.safe_stack(padded_seq, dim=0, device=device, dtype=dtype)
                padded_sequences.append(stacked_seq)
           
            latent_embeds = torch.stack(padded_sequences, dim=0)
           
            enhanced_embeds = torch.cat([
                inputs_embeds[:, :1, :],
                latent_embeds,
                inputs_embeds[:, 1:, :]
            ], dim=1)
           
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
                np.mean(info['num_trajectories']) if info['num_trajectories'] else 0
                for info in batch_trajectory_info
            ])
            result['avg_complexity'] = np.mean([
                np.mean(info['complexity_scores']) if info['complexity_scores'] else 0
                for info in batch_trajectory_info
            ])
            result['avg_diversity'] = np.mean([
                np.mean(info['diversity_scores']) if info['diversity_scores'] else 0
                for info in batch_trajectory_info
            ])
            result['trajectory_length'] = np.mean([info['actual_steps'] for info in batch_trajectory_info])
       
        return result
# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_generalized_coconut(debug_mode=False):
    """Complete training function with fixed validation loop"""
   
    # Configuration
    num_epochs = 1 if debug_mode else 3
    batch_size = 1 if debug_mode else 2
    gradient_accumulation_steps = 1 if debug_mode else 16
    max_length = 256 if debug_mode else 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    # Model configuration
    MIN_TRAJECTORIES = 2
    MAX_TRAJECTORIES = 4 if not debug_mode else 3
    MAX_LATENT_STEPS = 4 if not debug_mode else 3
   
    # Temperature schedule
    INITIAL_TEMP = 1.5
    FINAL_TEMP = 0.5
   
    # Learning rates
    base_learning_rate = 2.5e-6
    navigator_lr = 5e-6
   
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
        'memory_usage': [],
        'grad_norms': [],
        'accuracy_by_trajectories': defaultdict(list),
        'accuracy_by_steps': defaultdict(list)
    }
   
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
   
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False
    )
    base_model.resize_token_embeddings(len(tokenizer))
   
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
    for i, layer in enumerate(model.base_model.model.layers):
        if i < 28:
            for param in layer.parameters():
                param.requires_grad = False
   
    model.to(device)
   
    # Optimizer setup
    navigator_params = list(model.navigator.parameters())
    navigator_param_ids = {id(p) for p in navigator_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in navigator_param_ids]
   
    optimizer = torch.optim.AdamW([
        {'params': navigator_params, 'lr': navigator_lr, 'weight_decay': 0.001},
        {'params': base_params, 'lr': base_learning_rate, 'weight_decay': 0.01}
    ], betas=(0.9, 0.95))
   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
   
    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="train")
    val_dataset = load_dataset("gsm8k", "main", split="test")
   
    global_step = 0
    best_val_accuracy = 0.0
   
    for epoch in range(num_epochs):
        temperature = INITIAL_TEMP - (INITIAL_TEMP - FINAL_TEMP) * (epoch / num_epochs)
        epoch_losses = []
        epoch_correct = 0
        epoch_total_checked = 0
        epoch_diversity = []
        epoch_successful = 0
        epoch_failed = 0
        epoch_skipped = 0
        epoch_trajectory_counts = []
        epoch_step_counts = []
        epoch_grad_norms = []
        specialization_patterns = defaultdict(list)
       
        dataset_size = min(len(dataset), 10 if debug_mode else 500)
        num_batches = dataset_size // batch_size
       
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
       
        optimizer.zero_grad()
       
        for batch_idx in progress_bar:
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min(batch_start_idx + batch_size, dataset_size)
            batch_items = [dataset[i] for i in range(batch_start_idx, batch_end_idx)]
           
            batch_losses = []
            batch_correct = 0
            batch_checked = 0
            batch_errors = 0
           
            for item_idx, item in enumerate(batch_items):
                question = item['question']
                answer_text = item['answer']
               
                train_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution: {answer_text}"
               
                try:
                    inputs = tokenizer(
                        train_prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length,
                        padding=False
                    ).to(device)
                   
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids'],
                        temperature=temperature,
                        return_trajectory_info=True,
                        force_reasoning=True
                    )
                   
                    if 'loss' not in outputs or outputs['loss'] is None or not torch.isfinite(outputs['loss']):
                        epoch_skipped += 1
                        continue
                   
                    num_trajectories_used = outputs.get('avg_trajectories', 0)
                    trajectory_length = outputs.get('trajectory_length', 0)
                    diversity_score = outputs.get('avg_diversity', 0)
                   
                    traj_info = outputs.get('trajectory_info', {})
                    if isinstance(traj_info, list):
                        traj_info = traj_info[0]
                   
                    if 'num_trajectories' in traj_info and traj_info['num_trajectories']:
                        epoch_trajectory_counts.append(traj_info['num_trajectories'][0])
                   
                    if 'actual_steps' in traj_info:
                        epoch_step_counts.append(traj_info['actual_steps'])
                   
                    if 'specialization_info' in traj_info:
                        for step_spec in traj_info['specialization_info']:
                            for traj_id, spec in step_spec.items():
                                specialization_patterns[traj_id].append(spec)
                   
                    # Check accuracy
                    is_correct = False
                    with torch.no_grad():
                        inference_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
                        inference_inputs = tokenizer(
                            inference_prompt,
                            return_tensors='pt',
                            truncation=True,
                            max_length=max_length // 2
                        ).to(device)
                       
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
                       
                        pred_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', response)
                        true_nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
                       
                        if pred_nums and true_nums:
                            pred = pred_nums[-1].replace(',', '')
                            true = true_nums[-1].replace(',', '')
                            try:
                                if abs(float(pred) - float(true)) < 0.01:
                                    is_correct = True
                            except:
                                pass
                   
                    batch_correct += is_correct
                    batch_checked += 1
                    epoch_correct += is_correct
                    epoch_total_checked += 1
                   
                    metrics['accuracy_by_trajectories'][int(num_trajectories_used)].append(is_correct)
                    metrics['accuracy_by_steps'][int(trajectory_length)].append(is_correct)
                   
                    epoch_diversity.append(diversity_score)
                   
                    loss = outputs['loss']
                    if diversity_score > 0:
                        diversity_reward = diversity_score * 0.1
                        loss = loss - diversity_reward
                   
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                   
                    # Track gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    epoch_grad_norms.append(grad_norm.item())
                   
                    batch_losses.append(loss.item())
                    epoch_losses.append(loss.item())
                    epoch_successful += 1
                   
                except Exception as e:
                    print(f"Error in batch {batch_idx}, item {item_idx}: {e}")
                    traceback.print_exc()
                    epoch_failed += 1
                    continue
           
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
           
            postfix = {
                'loss': np.mean(batch_losses) if batch_losses else 'N/A',
                'acc': batch_correct / batch_checked if batch_checked > 0 else 'N/A',
                'grad_norm': np.mean(epoch_grad_norms[-10:]) if epoch_grad_norms else 'N/A'
            }
            progress_bar.set_postfix(postfix)
       
        metrics['grad_norms'].extend(epoch_grad_norms)
       
        # Epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f" • Avg loss: {np.mean(epoch_losses):.4f}" if epoch_losses else "N/A")
        print(f" • Accuracy: {epoch_correct / epoch_total_checked:.2%}" if epoch_total_checked > 0 else "N/A")
        print(f" • Avg grad norm: {np.mean(epoch_grad_norms):.4f}" if epoch_grad_norms else "N/A")
        print(f" • Successful/Failed: {epoch_successful}/{epoch_failed}")
        print(f" • Avg trajectories: {np.mean(epoch_trajectory_counts):.2f}" if epoch_trajectory_counts else "N/A")
        print(f" • Avg steps: {np.mean(epoch_step_counts):.2f}" if epoch_step_counts else "N/A")
       
        # Complete Validation Loop
        if not debug_mode:
            model.eval()
            val_correct = 0
            val_total = min(30, len(val_dataset))
           
            print("\nRunning validation...")
            with torch.no_grad():
                for i in tqdm(range(val_total), desc="Validation"):
                    item = val_dataset[i]
                    question = item['question']
                    answer_text = item['answer']
                   
                    inference_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
                   
                    try:
                        inference_inputs = tokenizer(
                            inference_prompt,
                            return_tensors='pt',
                            truncation=True,
                            max_length=max_length // 2
                        ).to(device)
                       
                        generated = model.base_model.generate(
                            inference_inputs['input_ids'],
                            attention_mask=inference_inputs['attention_mask'],
                            max_new_tokens=100,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                       
                        response = tokenizer.decode(
                            generated[0][inference_inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True
                        )
                       
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
                    except Exception as e:
                        print(f"Validation error on item {i}: {e}")
                        continue
           
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            metrics['val_accuracy'].append(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy:.2%} ({val_correct}/{val_total})")
           
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"New best model saved with accuracy: {best_val_accuracy:.2%}")
           
            model.train()
       
        scheduler.step()
   
    return model, metrics
# ============================================================
# TESTING FUNCTION
# ============================================================
def test_single_sample(model, tokenizer, dataset, device):
    """Test model on a single sample with detailed output"""
    model.eval()
   
    item = dataset[0]
    question = item['question']
    answer = item['answer']
   
    print(f"Question: {question}")
    print(f"True Answer: {answer}")
   
    inference_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
   
    with torch.no_grad():
        inputs = tokenizer(
            inference_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(device)
       
        # Test trajectory navigation
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_trajectory_info=True,
            temperature=0.7
        )
       
        if 'trajectory_info' in outputs:
            traj_info = outputs['trajectory_info']
            print(f"\nTrajectory Analysis:")
            print(f" Steps taken: {traj_info.get('actual_steps', 0)}")
            print(f" Avg trajectories: {outputs.get('avg_trajectories', 0):.2f}")
            print(f" Avg complexity: {outputs.get('avg_complexity', 0):.2f}")
            print(f" Avg diversity: {outputs.get('avg_diversity', 0):.2f}")
       
        # Generate answer
        generated = model.base_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
       
        response = tokenizer.decode(
            generated[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
       
        print(f"\nGenerated Answer: {response}")
if __name__ == "__main__":
    debug = '--debug' in sys.argv
    test_only = '--test' in sys.argv
   
    if test_only:
        # Test mode - load and test model
        print("Loading model for testing...")
       
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
        tokenizer.add_special_tokens(special_tokens)
       
        base_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B-Instruct',
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_cache=False
        )
        base_model.resize_token_embeddings(len(tokenizer))
       
        model = GeneralizedMultiTrajectoryCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
            start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
            end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
            eos_token_id=tokenizer.eos_token_id,
            hidden_size=4096,
            reasoning_dim=256,
            min_trajectories=2,
            max_trajectories=4,
            max_latent_steps=4,
            dropout_rate=0.1
        )
       
        # Load checkpoint if available
        checkpoint_path = 'checkpoints/best_model.pt'
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
       
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
       
        dataset = load_dataset("gsm8k", "main", split="test")
        test_single_sample(model, tokenizer, dataset, device)
    else:
        # Training mode
        print(f"Starting training in {'debug' if debug else 'normal'} mode...")
        model, metrics = train_generalized_coconut(debug_mode=debug)
       
        # Save final metrics
        with open('training_metrics.json', 'w') as f:
            json.dump({k: v for k, v in metrics.items() if not isinstance(v, defaultdict)}, f, indent=2)
       
        print("\nTraining complete!")
        print(f"Final metrics saved to training_metrics.json")
