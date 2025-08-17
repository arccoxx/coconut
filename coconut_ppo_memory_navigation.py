"""
Enhanced COCONUT with PPO - Fixed to handle training without labels mismatch
Integrated memory bank for state retrieval and storage during navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ContinuousReasoningNavigator(nn.Module):
    """Memory-efficient continuous reasoning navigator with dtype support and integrated memory bank"""
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        memory_size: int = 500,
        top_k_memory: int = 3,  # Number of top similar states to retrieve
        fusion_weight: float = 0.5,  # Weight for fusing retrieved directions
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.top_k_memory = top_k_memory
        self.fusion_weight = fusion_weight
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        
        # Efficient projection layers
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, reasoning_dim)
        )
        self.thought_projection = nn.Sequential(
            nn.Linear(reasoning_dim, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        
        # Policy heads
        self.continue_head = nn.Linear(reasoning_dim, 2)
        self.direction_head = nn.Linear(reasoning_dim, reasoning_dim)
        self.step_size_head = nn.Linear(reasoning_dim, 1)
        self.value_head = nn.Linear(reasoning_dim, 1)
        
        # Memory bank system
        self.register_buffer('memory_bank', torch.zeros(memory_size, reasoning_dim))
        self.register_buffer('memory_values', torch.full((memory_size,), -float('inf')))  # Initialize with low values
        self.memory_ptr = 0
        self.memory_filled = False  # Flag to indicate if bank has useful data
        
        # Move to device and dtype if specified
        if device:
            self.to(device)
        if dtype:
            self.to(dtype)
    
    def _retrieve_from_memory(self, reasoning_state: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve top-k similar states from memory bank based on cosine similarity"""
        if not self.memory_filled or self.top_k_memory <= 0:
            return None
            
        # Compute cosine similarity (batch-friendly)
        if reasoning_state.dim() == 1:
            reasoning_state = reasoning_state.unsqueeze(0)
        
        # Normalize for cosine
        norm_state = F.normalize(reasoning_state, p=2, dim=-1)
        norm_bank = F.normalize(self.memory_bank, p=2, dim=-1)
        
        similarities = torch.matmul(norm_state, norm_bank.t()).squeeze(0)  # [memory_size]
        
        # Get top-k indices based on similarity, weighted by values
        weighted_sims = similarities * (self.memory_values + 1e-8)  # Avoid div by zero
        top_k_indices = torch.topk(weighted_sims, k=min(self.top_k_memory, len(similarities)), sorted=False).indices
        
        if len(top_k_indices) == 0:
            return None
            
        # Retrieve and average the top-k states
        retrieved_states = self.memory_bank[top_k_indices]
        avg_retrieved = retrieved_states.mean(dim=0)
        
        return avg_retrieved
    
    def _write_to_memory(self, position: torch.Tensor, value: torch.Tensor):
        """Write new position and value to memory bank (circular overwrite)"""
        if position.dim() > 1:
            position = position.mean(dim=0)  # Average if batched
            
        self.memory_bank[self.memory_ptr] = position.detach()
        self.memory_values[self.memory_ptr] = value.detach().mean() if value.dim() > 0 else value.detach()
        self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank.size(0)
        
        # Mark as filled once we've written at least top_k entries
        if self.memory_ptr >= self.top_k_memory:
            self.memory_filled = True
    
    def navigate(self, state: torch.Tensor, return_policy_info: bool = True, generate_multiples: int = 1) -> Dict[str, torch.Tensor]:
        """Navigate through reasoning space with memory integration - handles both single and batch inputs
        Supports generating multiple trajectories by sampling from memory seeds.
        """
        # Ensure state is on the same device and dtype
        state = state.to(self.continue_head.weight.device)
        state = state.to(self.continue_head.weight.dtype)
        
        # Handle both single and batch inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = state.shape[0]
        
        # Project to reasoning space
        reasoning_state = self.state_projection(state)
        
        # Retrieve from memory and fuse if available
        retrieved = self._retrieve_from_memory(reasoning_state)
        if retrieved is not None:
            # Fuse retrieved average into current state
            reasoning_state = (1 - self.fusion_weight) * reasoning_state + self.fusion_weight * retrieved.unsqueeze(0)
        
        # Get policy outputs
        continue_logits = self.continue_head(reasoning_state)
        continue_probs = F.softmax(continue_logits, dim=-1)
        continue_dist = torch.distributions.Categorical(continue_probs)
        continue_action = continue_dist.sample()
        
        # Generate direction vector
        direction_raw = self.direction_head(reasoning_state)
        direction = F.normalize(direction_raw, p=2, dim=-1)
        
        # Step size
        step_size = torch.sigmoid(self.step_size_head(reasoning_state)) * 2.0
        
        # Value estimate
        value = self.value_head(reasoning_state)
        
        # Move in reasoning space
        next_position = reasoning_state + step_size * direction
        
        # Generate thought
        latent_thought = self.thought_projection(next_position)
        
        # Write to memory after computation
        self._write_to_memory(next_position, value)
        
        # Handle stop condition
        if single_input:
            stop_condition = continue_action.item() == 1
        else:
            stop_condition = (continue_action == 1)
        
        result = {
            'latent_thought': latent_thought.squeeze(0) if single_input else latent_thought,
            'stop': stop_condition,
            'position': next_position.squeeze(0) if single_input else next_position
        }
        
        if return_policy_info:
            result.update({
                'action': continue_action.squeeze(0) if single_input else continue_action,
                'log_prob': continue_dist.log_prob(continue_action).squeeze(0) if single_input else continue_dist.log_prob(continue_action),
                'value': value.squeeze() if single_input else value.squeeze(-1),
                'entropy': continue_dist.entropy().squeeze(0) if single_input else continue_dist.entropy()
            })
        
        # Generate multiples if requested (by sampling memory seeds for parallel navigations)
        if generate_multiples > 1:
            multiple_results = [result]  # Include the primary one
            for _ in range(1, generate_multiples):
                # Sample a memory state as seed for diversity
                if self.memory_filled:
                    mem_idx = torch.randint(0, self.memory_bank.size(0), (1,)).item()
                    seed_state = self.memory_bank[mem_idx].unsqueeze(0)
                else:
                    seed_state = reasoning_state.clone()  # Fallback to current
                
                # Recursive call with seeded state
                sub_result = self.navigate(seed_state.squeeze(0), return_policy_info=return_policy_info)
                multiple_results.append(sub_result)
            
            result['multiple_trajectories'] = multiple_results
        
        return result

class CoconutPPO(nn.Module):
    """COCONUT with PPO-based adaptive reasoning for Llama 3"""
    
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        hidden_size: int = 4096,
        reasoning_dim: int = 256,
        max_latent_steps: int = 6,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.max_latent_steps = max_latent_steps
        
        # Token IDs
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        
        # Get device and dtype from base model
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        
        # Reasoning navigator - create with same device and dtype as base model
        self.navigator = ContinuousReasoningNavigator(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
            device=device,
            dtype=dtype
        )
        
        # Enable gradient checkpointing
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trajectory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with PPO trajectory collection - handles batches properly"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Get initial embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Initialize trajectory storage for each batch item
        batch_trajectories = []
        batch_latent_sequences = []
        
        # Process each item in batch separately for trajectory collection
        for b in range(batch_size):
            trajectory = {
                'states': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'latent_embeds': []
            }
            
            # Get initial hidden state for this batch item
            with torch.no_grad():
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[b:b+1],
                    attention_mask=attention_mask[b:b+1] if attention_mask is not None else None,
                    output_hidden_states=True,
                    use_cache=False
                )
            
            current_state = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
            
            # Navigate through reasoning space
            latent_sequence = []
            for step in range(self.max_latent_steps):
                nav_output = self.navigator.navigate(current_state, return_policy_info=True)
                
                # Store trajectory info
                if return_trajectory:
                    trajectory['states'].append(current_state.detach())
                    trajectory['actions'].append(nav_output['action'])
                    trajectory['log_probs'].append(nav_output['log_prob'])
                    trajectory['values'].append(nav_output['value'])
                
                # Check stopping condition (now handles both single and batch)
                if isinstance(nav_output['stop'], bool):
                    should_stop = nav_output['stop']
                else:
                    should_stop = nav_output['stop'].item() if nav_output['stop'].numel() == 1 else False
                
                if should_stop:
                    break
                
                # Generate latent thought
                latent_thought = nav_output['latent_thought']
                latent_sequence.append(latent_thought)
                
                if return_trajectory:
                    trajectory['latent_embeds'].append(latent_thought.detach())
                
                # Update state
                current_state = latent_thought
            
            batch_trajectories.append(trajectory)
            batch_latent_sequences.append(latent_sequence)
        
        # Process full batch through model with latent thoughts
        max_latent_len = max(len(seq) for seq in batch_latent_sequences) if batch_latent_sequences else 0
        
        if max_latent_len > 0 and labels is not None:
            # When we have labels, we need to adjust them for the latent thoughts
            # Insert -100 (ignore index) for latent thought positions
            batch_size, seq_len = labels.shape
            
            # Create new labels with space for latent thoughts
            new_labels = torch.full(
                (batch_size, seq_len + max_latent_len),
                -100,  # Ignore index for cross entropy
                dtype=labels.dtype,
                device=labels.device
            )
            
            # Copy original labels to correct positions
            new_labels[:, 0] = labels[:, 0]  # First token
            new_labels[:, max_latent_len + 1:] = labels[:, 1:]  # Rest after latent
            
            labels = new_labels
        
        if max_latent_len > 0:
            # Pad latent sequences to same length
            padded_latent_sequences = []
            for seq in batch_latent_sequences:
                if len(seq) == 0:
                    # Create padding with correct shape
                    dummy_thought = torch.zeros(self.hidden_size, device=device, dtype=inputs_embeds.dtype)
                    padding = [dummy_thought for _ in range(max_latent_len)]
                    padded_seq = padding
                elif len(seq) < max_latent_len:
                    # Pad with zeros of the same shape
                    padding = [torch.zeros_like(seq[0]) for _ in range(max_latent_len - len(seq))]
                    padded_seq = seq + padding
                else:
                    padded_seq = seq
                
                # Stack them to create shape [seq_len, hidden_size]
                stacked_seq = torch.stack(padded_seq, dim=0)
                padded_latent_sequences.append(stacked_seq)
            
            # Stack for batch processing - shape will be [batch_size, seq_len, hidden_size]
            latent_embeds = torch.stack(padded_latent_sequences, dim=0)
            
            # Insert latent thoughts into sequence
            enhanced_embeds = torch.cat([
                inputs_embeds[:, :1, :],  # Start token
                latent_embeds,             # Latent thoughts
                inputs_embeds[:, 1:, :]    # Rest of sequence
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
            
            # Forward through model
            outputs = self.base_model(
                inputs_embeds=enhanced_embeds,
                attention_mask=enhanced_mask,
                labels=labels
            )
        else:
            # No latent thoughts, use original inputs
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        result = {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }
        
        if return_trajectory:
            # Return first trajectory or empty one if no trajectories
            result['trajectory'] = batch_trajectories[0] if batch_trajectories else {
                'states': [], 'actions': [], 'log_probs': [], 'values': [], 'latent_embeds': []
            }
            
        return result
