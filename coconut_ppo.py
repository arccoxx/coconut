"""
Enhanced COCONUT with PPO and continuous reasoning navigation
Optimized for Llama 3-8B on A100 40GB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaForCausalLM
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CoconutPPO(nn.Module):
    """COCONUT with PPO-based adaptive reasoning for Llama 3"""
    
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        hidden_size: int = 4096,  # Llama 3-8B hidden size
        reasoning_dim: int = 256,  # Reduced for memory efficiency
        max_latent_steps: int = 6,  # Reduced for A100 memory
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
        
        # Reasoning navigator (continuous)
        self.navigator = ContinuousReasoningNavigator(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
            memory_efficient=True  # For A100 constraints
        )
        
        # PPO components
        self.policy_network = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.LayerNorm(reasoning_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.LayerNorm(reasoning_dim // 2),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1)
        )
        
        # Gradient checkpointing for memory efficiency
        self.base_model.gradient_checkpointing_enable()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trajectory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with PPO trajectory collection
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Get initial embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Initialize trajectory storage for PPO
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'latent_embeds': []
        }
        
        # Get initial hidden state
        with torch.no_grad():  # Save memory during trajectory generation
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False  # Disable KV cache to save memory
            )
        
        current_state = outputs.hidden_states[-1].mean(dim=1)
        
        # Navigate through reasoning space
        latent_sequence = []
        for step in range(self.max_latent_steps):
            # Get navigation decision from policy
            nav_output = self.navigator.navigate(
                current_state,
                return_policy_info=True
            )
            
            # Store trajectory info
            trajectory['states'].append(current_state.detach())
            trajectory['actions'].append(nav_output['action'])
            trajectory['log_probs'].append(nav_output['log_prob'])
            trajectory['values'].append(nav_output['value'])
            
            # Check stopping condition
            if nav_output['stop']:
                break
            
            # Generate latent thought
            latent_thought = nav_output['latent_thought']
            latent_sequence.append(latent_thought)
            trajectory['latent_embeds'].append(latent_thought.detach())
            
            # Update state
            current_state = latent_thought
        
        # Combine latent thoughts with original input
        if latent_sequence:
            enhanced_embeds = self._integrate_latent_thoughts(
                inputs_embeds, latent_sequence, attention_mask
            )
            
            # Forward through model with enhanced embeddings
            outputs = self.base_model(
                inputs_embeds=enhanced_embeds,
                attention_mask=self._update_attention_mask(
                    attention_mask, len(latent_sequence)
                ),
                labels=labels
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        # Prepare return dict
        result = {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }
        
        if return_trajectory:
            result['trajectory'] = trajectory
            
        return result
    
    def _integrate_latent_thoughts(
        self,
        original_embeds: torch.Tensor,
        latent_sequence: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Integrate latent thoughts into embedding sequence"""
        # Stack latent thoughts
        latent_embeds = torch.stack(latent_sequence, dim=1)
        
        # Insert latent thoughts after start token
        combined = torch.cat([
            original_embeds[:, :1, :],  # Start token
            latent_embeds,               # Latent thoughts
            original_embeds[:, 1:, :]   # Rest of sequence
        ], dim=1)
        
        return combined
    
    def _update_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        num_latent: int
    ) -> torch.Tensor:
        """Update attention mask for latent thoughts"""
        if attention_mask is None:
            return None
            
        batch_size = attention_mask.shape[0]
        device = attention_mask.device
        
        # Add attention for latent thoughts
        latent_attention = torch.ones(
            batch_size, num_latent, 
            dtype=attention_mask.dtype,
            device=device
        )
        
        return torch.cat([
            attention_mask[:, :1],
            latent_attention,
            attention_mask[:, 1:]
        ], dim=1)

class ContinuousReasoningNavigator(nn.Module):
    """
    Memory-efficient continuous reasoning navigator for A100
    """
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        memory_size: int = 500,  # Reduced for memory
        memory_efficient: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.memory_efficient = memory_efficient
        
        # Efficient projection layers
        if memory_efficient:
            # Use smaller intermediate dimensions
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
        else:
            self.state_projection = nn.Linear(hidden_size, reasoning_dim)
            self.thought_projection = nn.Linear(reasoning_dim, hidden_size)
        
        # Policy heads
        self.continue_head = nn.Linear(reasoning_dim, 2)
        self.direction_head = nn.Linear(reasoning_dim, reasoning_dim)
        self.step_size_head = nn.Linear(reasoning_dim, 1)
        self.value_head = nn.Linear(reasoning_dim, 1)
        
        # Small memory bank
        self.register_buffer('memory_bank', torch.zeros(memory_size, reasoning_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size))
        self.memory_ptr = 0
        
    def navigate(
        self,
        state: torch.Tensor,
        return_policy_info: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Navigate through reasoning space"""
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
        
        result = {
            'latent_thought': latent_thought,
            'stop': continue_action.item() == 1,
            'position': next_position
        }
        
        if return_policy_info:
            result.update({
                'action': continue_action,
                'log_prob': continue_dist.log_prob(continue_action),
                'value': value.squeeze(),
                'entropy': continue_dist.entropy()
            })
        
        return result
