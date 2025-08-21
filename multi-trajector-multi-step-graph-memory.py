#!/usr/bin/env python3
"""
Enhanced Multi-Trajectory COCONUT with Graph Neural Network PPO Policy
======================================================================
Clean, debugged version with proper dtype handling and full evaluation metrics.
Reverted training logic to stable version while preserving evaluation bug fixes.

Architecture: Hybrid GAT-Transformer with PNA Aggregation
Author: Enhanced by Claude
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
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check environment
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

if 'get_ipython' in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TrajectoryState:
    """Encapsulates trajectory state"""
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

# ============================================================
# SCATTER OPERATIONS FOR GNN
# ============================================================

class torch_scatter:
    """Basic scatter operations for GNN without external dependencies"""
    
    @staticmethod
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim_size is None:
            if index.numel() > 0:
                dim_size = index.max().item() + 1
            else:
                dim_size = 0
        if dim_size <= 0:
            return torch.zeros(0, *src.shape[1:], dtype=src.dtype, device=src.device)
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.index_add_(dim, index, src)
        return out
    
    @staticmethod
    def scatter_mean(src, index, dim=0, dim_size=None):
        if dim_size is None:
            if index.numel() > 0:
                dim_size = index.max().item() + 1
            else:
                dim_size = 0
        if dim_size <= 0:
            return torch.zeros(0, *src.shape[1:], dtype=src.dtype, device=src.device)
        sum_out = torch_scatter.scatter_add(src, index, dim, dim_size)
        count = torch_scatter.scatter_add(torch.ones_like(src), index, dim, dim_size)
        return sum_out / count.clamp(min=1)
    
    @staticmethod
    def scatter_max(src, index, dim=0, dim_size=None):
        if dim_size is None:
            if index.numel() > 0:
                dim_size = index.max().item() + 1
            else:
                dim_size = 0
        if dim_size <= 0:
            return torch.zeros(0, *src.shape[1:], dtype=src.dtype, device=src.device), torch.zeros(0, *src.shape[1:], dtype=torch.long, device=src.device)
        out = torch.full((dim_size, *src.shape[1:]), float('-inf'), dtype=src.dtype, device=src.device)
        unique_indices = torch.unique(index)
        for idx in unique_indices:
            out[idx] = torch.max(src[index == idx], dim=0)[0]
        return out, torch.zeros_like(out, dtype=torch.long)
    
    class composite:
        @staticmethod
        def scatter_softmax(src, index, dim=0):
            if index.numel() == 0:
                return torch.empty_like(src)
            src_float = src.float()
            max_value_per_index = torch_scatter.scatter_max(src_float, index, dim)[0]
            max_per_src_element = max_value_per_index[index]
            recentered_scores = src_float - max_per_src_element
            exp_scores = recentered_scores.exp()
            sum_exp_scores = torch_scatter.scatter_add(exp_scores, index, dim)
            sum_per_src_element = sum_exp_scores[index]
            result = exp_scores / sum_per_src_element.clamp(min=1e-10)
            return result.to(src.dtype)

# ============================================================
# GRAPH NEURAL NETWORK COMPONENTS
# ============================================================

class GraphAttentionLayer(nn.Module):
    """Enhanced Graph Attention Layer with edge features and proper dtype handling"""
    
    def __init__(self, in_features: int, out_features: int, edge_features: int = 0,
                 heads: int = 8, dropout: float = 0.1, concat: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout_val = dropout
        self.device = device if device else torch.device('cpu')
        self.dtype = dtype if dtype else torch.float32
        
        self.W = nn.Linear(in_features, heads * out_features, bias=False, device=device).to(dtype)
        self.a_src = nn.Parameter(torch.zeros(heads, out_features, device=device, dtype=dtype))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_features, device=device, dtype=dtype))
        
        if edge_features > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_features, heads * out_features, device=device),
                nn.ReLU(),
            ).to(dtype)
        else:
            self.edge_encoder = None
            
        self.bias = nn.Parameter(torch.zeros(heads * out_features if concat else out_features, device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.a_src, gain=gain)
        nn.init.xavier_uniform_(self.a_dst, gain=gain)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.size(0)
        if N == 0:
            return torch.empty(0, self.heads * self.out_features if self.concat else self.out_features, device=x.device, dtype=x.dtype)

        x = x.to(self.dtype)
        h = self.W(x).view(N, self.heads, self.out_features)
        
        src, dst = edge_index
        alpha_src = (h[src] * self.a_src).sum(dim=-1)
        alpha_dst = (h[dst] * self.a_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst
        
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr.to(self.dtype)).view(-1, self.heads, self.out_features)
            alpha = alpha + (edge_attr_encoded * h[dst]).sum(dim=-1)
            
        alpha = self.leakyrelu(alpha)
        alpha = torch_scatter.composite.scatter_softmax(alpha, dst, dim=0)
        alpha = self.dropout(alpha)
        
        # Message passing
        out = h[src] * alpha.unsqueeze(-1)
        
        # FIX: Reverted to correct aggregation logic
        N_heads = out.size(1)
        N_features = out.size(2)
        out_flat = out.view(-1, N_features)
        dst_expanded = dst.unsqueeze(1).expand(-1, N_heads).flatten()
        
        aggregated = torch.zeros(N, N_heads, N_features, dtype=out.dtype, device=out.device)
        aggregated.view(-1, N_features).index_add_(0, dst_expanded, out_flat)

        if self.concat:
            out = aggregated.view(N, self.heads * self.out_features)
        else:
            out = aggregated.mean(dim=1)
            
        return out + self.bias

class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer for global reasoning with proper dtype handling"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1, device=None, dtype=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True, device=device).to(dtype)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim, device=device),
            nn.Dropout(dropout)
        ).to(dtype)
        self.norm1 = nn.LayerNorm(hidden_dim, device=device).to(dtype)
        self.norm2 = nn.LayerNorm(hidden_dim, device=device).to(dtype)
        self.dtype = dtype
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.to(self.dtype)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class GraphNeuralPPOPolicy(nn.Module):
    """Advanced GNN-based PPO Policy Network for trajectory navigation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_gnn_layers: int = 3, num_transformer_layers: int = 2,
                 memory_size: int = 200, k_neighbors: int = 10,
                 device=None, dtype=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.device = device if device else torch.device('cpu')
        self.dtype = dtype if dtype else torch.float32
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device), nn.ReLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, hidden_dim, device=device)
        ).to(dtype)
        
        self.memory_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, device=device), nn.ReLU(),
            nn.LayerNorm(hidden_dim, device=device),
            nn.Linear(hidden_dim, hidden_dim, device=device)
        ).to(dtype)
        
        num_heads = 8
        self.gat_layers = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(num_gnn_layers):
            is_last_layer = (i == num_gnn_layers - 1)
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim, out_features=hidden_dim, edge_features=3,
                    heads=num_heads, concat=(not is_last_layer), device=device, dtype=dtype
                )
            )
            if not is_last_layer:
                in_dim = hidden_dim * num_heads
        
        self.transformer_layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, num_heads=8, dropout=0.1, device=device, dtype=dtype)
             for _ in range(num_transformer_layers)]
        )
        
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1, device=device), nn.Sigmoid()
        ).to(dtype)
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim, device=device)
        ).to(dtype)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, 1, device=device)
        ).to(dtype)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2, device=device), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3, device=device)
        ).to(dtype)
        
    def build_memory_graph(self, state: torch.Tensor, memory_bank: torch.Tensor,
                           memory_values: torch.Tensor, memory_timestamps: torch.Tensor) -> Tuple:
        state_h = self.state_encoder(state.to(self.dtype))
        memory_h = self.memory_encoder(memory_bank.to(self.dtype))
        
        batch_size = state_h.size(0)
        distances = torch.cdist(state_h.float(), memory_h.float(), p=2).to(self.dtype)
        
        k = min(self.k_neighbors, memory_bank.size(0))
        knn_distances, knn_indices = torch.topk(distances, k, largest=False, dim=-1)
        
        src_nodes, dst_nodes = [], []
        edge_features = []
        
        for i in range(batch_size):
            src_nodes.extend([i] * k)
            dst_nodes.extend(knn_indices[i] + batch_size)
            
            mem_vals = memory_values[knn_indices[i]]
            mem_ts = memory_timestamps[knn_indices[i]].to(self.dtype)
            edge_features.append(torch.stack([knn_distances[i], mem_vals, mem_ts], dim=-1))

        edge_index = torch.tensor([src_nodes, dst_nodes], device=self.device, dtype=torch.long)
        edge_attr = torch.cat(edge_features, dim=0)
        
        node_features = torch.cat([state_h, memory_h], dim=0)
        return node_features, edge_index, edge_attr

    def forward(self, state: torch.Tensor, memory_bank: Optional[torch.Tensor] = None,
                memory_values: Optional[torch.Tensor] = None,
                memory_timestamps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        state_batch = state.unsqueeze(0) if state.dim() == 1 else state
        
        if memory_bank is None or memory_bank.size(0) == 0:
            state_h = self.state_encoder(state_batch)
            dummy_features = torch.cat([state_h, state_h, state_h], dim=-1)
            return self.policy_head(dummy_features), self.value_head(dummy_features)

        outputs = [self._forward_single(s, memory_bank, memory_values, memory_timestamps) for s in state_batch]
        action_logits = torch.cat([o[0] for o in outputs], dim=0)
        values = torch.cat([o[1] for o in outputs], dim=0)
        return action_logits, values

    def _forward_single(self, state: torch.Tensor, memory_bank: torch.Tensor,
                        memory_values: torch.Tensor, memory_timestamps: torch.Tensor):
        state_batch = state.unsqueeze(0)
        node_features, edge_index, edge_attr = self.build_memory_graph(
            state_batch, memory_bank, memory_values, memory_timestamps
        )
        edge_attr = self.edge_encoder(edge_attr)
        
        h = node_features
        for layer in self.gat_layers:
            h = F.relu(layer(h, edge_index, edge_attr))
        
        for layer in self.transformer_layers:
            h = layer(h.unsqueeze(0)).squeeze(0)
        
        traj_repr = h[:1]
        mem_repr = h[1:].mean(dim=0, keepdim=True)
        
        attention_weights = self.global_attention(h)
        global_repr = (h * attention_weights).sum(dim=0, keepdim=True)
        
        combined_repr = torch.cat([traj_repr, mem_repr, global_repr], dim=-1)
        
        return self.policy_head(combined_repr), self.value_head(combined_repr)

class GNNEnhancedMultiTrajectoryNavigator(nn.Module):
    def __init__(self, hidden_size: int, reasoning_dim: int = 256, max_trajectories: int = 5,
                 shared_memory_size: int = 200, dropout_rate: float = 0.1,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.max_trajectories = max_trajectories
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.bfloat16
        
        self.gnn_policy = GraphNeuralPPOPolicy(
            state_dim=reasoning_dim, action_dim=reasoning_dim, hidden_dim=reasoning_dim,
            device=device, dtype=dtype
        )
        self.state_projection = nn.Linear(hidden_size, reasoning_dim, device=self.device).to(self.dtype)
        self.thought_projection = nn.Linear(reasoning_dim, hidden_size, device=self.device).to(self.dtype)
        
        self.memory_bank = torch.zeros(shared_memory_size, reasoning_dim, device=device, dtype=dtype)
        self.memory_values = torch.zeros(shared_memory_size, device=device, dtype=dtype)
        self.memory_timestamps = torch.zeros(shared_memory_size, device=device, dtype=torch.long)
        self.memory_ptr = 0
        self.total_writes = 0
        
        self.trajectory_heads = nn.ModuleList([
            nn.ModuleDict({
                'continue': nn.Linear(reasoning_dim, 2, device=self.device).to(self.dtype),
                'confidence': nn.Linear(reasoning_dim, 1, device=self.device).to(self.dtype),
            }) for _ in range(max_trajectories)
        ])
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=reasoning_dim, num_heads=4, dropout=dropout_rate, batch_first=True, device=self.device
        ).to(self.dtype)

    def update_memory(self, position: torch.Tensor, value: float):
        with torch.no_grad():
            pos = position.squeeze().to(self.dtype)
            val = value.item() if isinstance(value, torch.Tensor) else value
            self.memory_bank[self.memory_ptr] = pos
            self.memory_values[self.memory_ptr] = val
            self.memory_timestamps[self.memory_ptr] = self.total_writes
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank.size(0)
            self.total_writes += 1

    def navigate_with_gnn(self, state: torch.Tensor, num_trajectories: int, temperature: float = 1.0) -> Dict[str, Any]:
        state_projected = self.state_projection(state.to(self.dtype))
        
        valid_mem_size = self.total_writes if self.total_writes < self.memory_bank.size(0) else self.memory_bank.size(0)
        valid_memory = self.memory_bank[:valid_mem_size] if valid_mem_size > 0 else None

        action_logits, values = self.gnn_policy(
            state_projected.expand(num_trajectories, -1),
            valid_memory,
            self.memory_values[:valid_mem_size] if valid_mem_size > 0 else None,
            self.memory_timestamps[:valid_mem_size] if valid_mem_size > 0 else None
        )
        
        trajectory_outputs = []
        for i in range(num_trajectories):
            pos = action_logits[i]
            if self.training:
                pos = torch.distributions.Normal(pos, temperature).sample()
            
            heads = self.trajectory_heads[i % self.max_trajectories]
            continue_probs = F.softmax(heads['continue'](pos.unsqueeze(0)), dim=-1)
            stop_vote = torch.argmax(continue_probs).item() == 1
            
            self.update_memory(pos, values[i])
            trajectory_outputs.append({
                'thought': self.thought_projection(pos),
                'position': pos,
                'stop_vote': stop_vote,
                'value': values[i],
            })

        positions = torch.stack([out['position'] for out in trajectory_outputs])
        attended_positions, _ = self.cross_attention(positions.unsqueeze(0), positions.unsqueeze(0), positions.unsqueeze(0))
        
        for i, out in enumerate(trajectory_outputs):
            out['position'] = attended_positions.squeeze(0)[i]

        ensemble_thought = torch.stack([out['thought'] for out in trajectory_outputs]).mean(dim=0)
        ensemble_stop = sum(out['stop_vote'] for out in trajectory_outputs) > num_trajectories / 2
        
        return {
            'ensemble_thought': ensemble_thought,
            'ensemble_stop': torch.tensor(ensemble_stop, device=self.device),
            'num_trajectories': num_trajectories,
            'memory_usage': valid_mem_size
        }

class EnhancedCoconutWithGNN(nn.Module):
    def __init__(self, base_model: LlamaForCausalLM, hidden_size: int, reasoning_dim: int, max_trajectories: int, max_latent_steps: int, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.max_latent_steps = max_latent_steps
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        
        self.navigator = GNNEnhancedMultiTrajectoryNavigator(
            hidden_size=hidden_size, reasoning_dim=reasoning_dim, device=device, dtype=dtype, max_trajectories=max_trajectories
        )
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        ).to(device).to(dtype)
        self.traj_count_net = nn.Sequential(
            nn.Linear(hidden_size + 1, max_trajectories - 2 + 1), nn.Softmax(dim=-1)
        ).to(device).to(dtype)
        
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        with torch.no_grad():
            outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
            initial_state = outputs.hidden_states[-1][:, -1, :]
        
        complexity_score = self.complexity_analyzer(initial_state)
        traj_probs = self.traj_count_net(torch.cat([initial_state, complexity_score], dim=-1))
        num_trajectories = 2 + torch.argmax(traj_probs, dim=-1)
        
        all_thoughts = []
        for b in range(batch_size):
            current_state = initial_state[b]
            thoughts = []
            for _ in range(self.max_latent_steps):
                nav_output = self.navigator.navigate_with_gnn(current_state, num_trajectories[b].item())
                thoughts.append(nav_output['ensemble_thought'])
                current_state = nav_output['ensemble_thought']
                if nav_output['ensemble_stop']:
                    break
            all_thoughts.append(torch.stack(thoughts))

        max_len = max(len(t) for t in all_thoughts)
        padded_thoughts = torch.stack([F.pad(t, (0, 0, 0, max_len - len(t))) for t in all_thoughts])
        
        final_embeds = torch.cat([inputs_embeds, padded_thoughts], dim=1)
        final_mask = F.pad(attention_mask, (0, max_len), value=1)
        
        if labels is not None:
            labels = F.pad(labels, (0, max_len), value=-100)
            
        outputs = self.base_model(inputs_embeds=final_embeds, attention_mask=final_mask, labels=labels)
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'avg_trajectories': num_trajectories.float().mean().item(),
            'trajectory_length': np.mean([len(t) for t in all_thoughts])
        }

def extract_last_number(text: str) -> Optional[float]:
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return float(numbers[-1]) if numbers else None

def evaluate_model(model, tokenizer, val_dataset, device, debug_mode=False):
    model.eval()
    correct = 0
    total = min(len(val_dataset), 10 if debug_mode else 100)
    
    with torch.no_grad():
        for i in tqdm(range(total), desc="Evaluating"):
            item = val_dataset[i]
            prompt = f"Question: {item['question']}\nLet's solve this step by step.\n\nSolution:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            generated_ids = model.base_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            pred_answer = extract_last_number(decoded_output)
            true_answer = extract_last_number(item['answer'])
            
            if pred_answer is not None and true_answer is not None and abs(pred_answer - true_answer) < 1e-3:
                correct += 1
                
    accuracy = correct / total
    model.train()
    return accuracy

def test_gnn_model():
    print("Testing GNN model components...")
    print("-" * 50)
    
    assert extract_last_number("The answer is #### 1,234.5") == 1234.5, "Answer extraction failed"
    print("   ✓ Answer extraction test passed")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    gnn_policy = GraphNeuralPPOPolicy(state_dim=256, action_dim=256, hidden_dim=256, device=device, dtype=dtype)
    
    state = torch.randn(256, device=device, dtype=dtype)
    memory_bank = torch.randn(10, 256, device=device, dtype=dtype)
    memory_values = torch.randn(10, device=device, dtype=dtype)
    memory_timestamps = torch.randint(0, 100, (10,), device=device)
    
    try:
        action_logits, value = gnn_policy(state, memory_bank, memory_values, memory_timestamps)
        assert action_logits.shape == (1, 256) and value.shape == (1, 1)
        print(f"   ✓ Single state test passed")
    except Exception as e:
        print(f"   ✗ Single state failed: {e}"); return False

    state_batch = torch.randn(2, 256, device=device, dtype=dtype)
    try:
        action_logits, value = gnn_policy(state_batch, memory_bank, memory_values, memory_timestamps)
        assert action_logits.shape == (2, 256) and value.shape == (2, 1)
        print(f"   ✓ Batched state test passed")
    except Exception as e:
        print(f"   ✗ Batched state failed: {e}"); return False
        
    print("="*50 + "\n✓ All tests passed successfully!\n" + "="*50)
    return True

def train_gnn_coconut(debug_mode=False):
    if not test_gnn_model():
        print("Model tests failed! Aborting training.")
        return None, {}
    
    config = {
        "num_epochs": 1 if debug_mode else 3,
        "batch_size": 1 if debug_mode else 2,
        "grad_accum": 1 if debug_mode else 16,
        "max_length": 256 if debug_mode else 512,
        "base_lr": 2.5e-6, "gnn_lr": 5e-6,
        "max_traj": 4, "max_steps": 4,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', pad_token='<|end_of_text|>')
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.bfloat16, device_map='auto'
    )
    
    model = EnhancedCoconutWithGNN(
        base_model=base_model, hidden_size=4096, reasoning_dim=256,
        max_trajectories=config['max_traj'], max_latent_steps=config['max_steps']
    )
    
    optimizer = torch.optim.AdamW([
        {'params': model.navigator.parameters(), 'lr': config['gnn_lr']},
        {'params': list(model.base_model.parameters()) + list(model.complexity_analyzer.parameters()) + list(model.traj_count_net.parameters()), 'lr': config['base_lr']}
    ])
    
    dataset = load_dataset("gsm8k", "main", split="train")
    val_dataset = load_dataset("gsm8k", "main", split="test")
    
    metrics = defaultdict(list)
    best_val_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        model.train()
        dataset_size = min(len(dataset), 10 if debug_mode else 500)
        progress_bar = tqdm(range(0, dataset_size, config['batch_size']), desc=f"Epoch {epoch+1}")
        
        for i in progress_bar:
            batch = [dataset[j] for j in range(i, min(i + config['batch_size'], dataset_size))]
            prompts = [f"Question: {item['question']}\nLet's solve this step by step.\n\nSolution: {item['answer']}" for item in batch]
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=config['max_length']).to(device)
            
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs['loss'] / config['grad_accum']
            loss.backward()
            
            if (i // config['batch_size'] + 1) % config['grad_accum'] == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                metrics['grad_norms'].append(grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()
            
            metrics['train_loss'].append(loss.item() * config['grad_accum'])
            progress_bar.set_postfix(loss=np.mean(metrics['train_loss'][-50:]))

        val_accuracy = evaluate_model(model, tokenizer, val_dataset, device, debug_mode)
        metrics['val_accuracy'].append(val_accuracy)
        print(f"\nEpoch {epoch+1} Summary: Train Loss={np.mean(metrics['train_loss'][-len(progress_bar):]):.4f}, Val Accuracy={val_accuracy:.4f}\n")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("New best validation accuracy! Saving model...")
            # Move the model's state_dict to CPU before saving
            cpu_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
            torch.save(cpu_state_dict, 'best_gnn_coconut_model.pt')

    return model, metrics

if __name__ == "__main__":
    debug = '--debug' in sys.argv
    model, metrics = train_gnn_coconut(debug_mode=debug)
    if model:
        with open('training_metrics_gnn.json', 'w') as f:
            json.dump({k: v for k, v in metrics.items()}, f, indent=2)
        print("Training complete! Metrics saved to training_metrics_gnn.json")
