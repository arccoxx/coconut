"""
Enhanced COCONUT with PPO - Fixed for batch processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ContinuousReasoningNavigator(nn.Module):
    """Memory-efficient continuous reasoning navigator with batch support"""
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        memory_size: int = 500,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.device = device
        
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
        
        # Small memory bank
        self.register_buffer('memory_bank', torch.zeros(memory_size, reasoning_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size))
        self.memory_ptr = 0
        
        # Move to device if specified
        if device:
            self.to(device)
    
    def navigate(self, state: torch.Tensor, return_policy_info: bool = True) -> Dict[str, torch.Tensor]:
        """Navigate through reasoning space - handles both single and batch inputs"""
        # Ensure state is on the same device
        state = state.to(self.continue_head.weight.device)
        
        # Handle both single and batch inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = state.shape[0]
        
        # Project to reasoning space
        reasoning_state = self.state_projection(state)
        
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
        
        # For batch processing, we need to handle stop condition differently
        if single_input:
            stop_condition = continue_action.item() == 1
        else:
            # For batch, return tensor of stop conditions
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
        
        # Get device from base model
        device = next(base_model.parameters()).device
        
        # Reasoning navigator - create on the same device as base model
        self.navigator = ContinuousReasoningNavigator(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
            device=device
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
        max_latent_len = max(len(seq) for seq in batch_latent_sequences)
        
        if max_latent_len > 0:
            # Pad latent sequences to same length
            padded_latent_sequences = []
            for seq in batch_latent_sequences:
                if len(seq) < max_latent_len:
                    # Pad with zeros
                    padding = [torch.zeros_like(seq[0]) for _ in range(max_latent_len - len(seq))]
                    padded_seq = seq + padding
                else:
                    padded_seq = seq
                padded_latent_sequences.append(torch.stack(padded_seq))
            
            # Stack for batch processing
            latent_embeds = torch.stack(padded_latent_sequences)
            
            # Insert latent thoughts into sequence
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
            
            # Forward through model
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
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }
        
        if return_trajectory:
            # For now, return first trajectory (can be extended to handle all)
            result['trajectory'] = batch_trajectories[0]
            
        return result
