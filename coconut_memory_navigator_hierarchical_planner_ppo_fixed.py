"""
FIXED COCONUT Model with Memory-Efficient Navigator and Hierarchical Planning
Fixes for test failures in IntegratedValueNetwork and label handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# PPO EXPERIENCE AND BUFFER (UNCHANGED)
# ============================================================

@dataclass
class PPOExperience:
    """Single experience for PPO training"""
    state: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    next_state: Optional[torch.Tensor] = None
    done: bool = False
    info: Dict[str, Any] = None

class PPOReplayBuffer:
    """PPO Replay Buffer with proper GAE computation"""
    
    def __init__(self, capacity: int = 2048, gamma: float = 0.99, gae_lambda: float = 0.95, device: str = 'cuda'):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.episode_buffer = []
        
    def add_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool = False,
        info: Dict = None
    ):
        """Add single step to episode buffer"""
        experience = PPOExperience(
            state=state.detach().cpu() if torch.is_tensor(state) else torch.tensor(state),
            action=action.detach().cpu() if torch.is_tensor(action) else torch.tensor(action),
            log_prob=log_prob.detach().cpu() if torch.is_tensor(log_prob) else torch.tensor(log_prob),
            value=value.detach().cpu() if torch.is_tensor(value) else torch.tensor(value),
            reward=reward,
            done=done,
            info=info or {}
        )
        self.episode_buffer.append(experience)
        
        if done:
            self.finish_episode()
    
    def finish_episode(self):
        """Complete episode and compute returns/advantages"""
        if not self.episode_buffer:
            return
        
        # Compute returns and advantages
        self._compute_returns_and_advantages()
        
        # Add to main buffer
        self.buffer.extend(self.episode_buffer)
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def _compute_returns_and_advantages(self):
        """Compute GAE returns and advantages"""
        if not self.episode_buffer:
            return
        
        # Extract values and rewards
        values = torch.tensor([exp.value for exp in self.episode_buffer])
        rewards = torch.tensor([exp.reward for exp in self.episode_buffer])
        
        # Initialize storage
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute backwards
        last_return = 0
        last_value = 0
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - self.episode_buffer[t].done
            
            returns[t] = rewards[t] + self.gamma * next_non_terminal * last_return
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            
            last_return = returns[t]
            last_advantage = advantages[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store in experiences
        for i, exp in enumerate(self.episode_buffer):
            exp.info['return'] = returns[i].item()
            exp.info['advantage'] = advantages[i].item()
    
    def sample_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample batch for training"""
        if len(self.buffer) < batch_size:
            return None
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        # Stack tensors
        batch = {
            'states': torch.stack([exp.state for exp in experiences]).to(self.device),
            'actions': torch.stack([exp.action for exp in experiences]).to(self.device),
            'old_log_probs': torch.stack([exp.log_prob for exp in experiences]).to(self.device),
            'advantages': torch.tensor([exp.info.get('advantage', 0) for exp in experiences], device=self.device),
            'returns': torch.tensor([exp.info.get('return', 0) for exp in experiences], device=self.device),
            'values': torch.stack([exp.value for exp in experiences]).to(self.device)
        }
        
        return batch
    
    def get_all_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get all data in batches for full buffer training"""
        if len(self.buffer) == 0:
            return []
        
        # Shuffle indices
        indices = np.random.permutation(len(self.buffer))
        
        batches = []
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            experiences = [self.buffer[i] for i in batch_indices]
            
            batch = {
                'states': torch.stack([exp.state for exp in experiences]).to(self.device),
                'actions': torch.stack([exp.action for exp in experiences]).to(self.device),
                'old_log_probs': torch.stack([exp.log_prob for exp in experiences]).to(self.device),
                'advantages': torch.tensor([exp.info.get('advantage', 0) for exp in experiences], device=self.device),
                'returns': torch.tensor([exp.info.get('return', 0) for exp in experiences], device=self.device),
                'values': torch.stack([exp.value for exp in experiences]).to(self.device)
            }
            batches.append(batch)
        
        return batches
    
    def clear(self):
        """Clear all buffers"""
        self.buffer.clear()
        self.episode_buffer = []
    
    def __len__(self):
        return len(self.buffer)

# ============================================================
# MEMORY-EFFICIENT CONTINUOUS REASONING NAVIGATOR (UNCHANGED)
# ============================================================

class MemoryEfficientNavigator(nn.Module):
    """Memory-efficient continuous reasoning navigator with value estimation"""
    
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        memory_size: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.memory_size = memory_size
        
        # State projection with normalization
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, reasoning_dim),
            nn.LayerNorm(reasoning_dim)
        )
        
        # Thought generation
        self.thought_projection = nn.Sequential(
            nn.Linear(reasoning_dim, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        
        # Policy heads
        self.continue_head = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 2)
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim * 2),
            nn.ReLU(),
            nn.Linear(reasoning_dim * 2, reasoning_dim)
        )
        
        self.step_size_head = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.ReLU(),
            nn.Linear(reasoning_dim // 2, 1)
        )
        
        # Memory bank for trajectory storage
        self.register_buffer('memory_bank', torch.zeros(memory_size, reasoning_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def update_memory(self, state: torch.Tensor, value: torch.Tensor):
        """Update memory bank with new state"""
        if self.training:
            with torch.no_grad():
                ptr = self.memory_ptr.item()
                self.memory_bank[ptr] = state.detach()
                self.memory_values[ptr] = value.detach()
                self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def navigate(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        return_policy_info: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Navigate through reasoning space"""
        # Handle batch dimensions
        single_input = state.dim() == 1
        if single_input:
            state = state.unsqueeze(0)
        
        # Project to reasoning space
        if state.shape[-1] != self.reasoning_dim:
            reasoning_state = self.state_projection(state)
        else:
            reasoning_state = state
        
        # Get policy outputs
        continue_logits = self.continue_head(reasoning_state)
        continue_probs = F.softmax(continue_logits, dim=-1)
        continue_dist = torch.distributions.Categorical(continue_probs)
        
        if deterministic:
            continue_action = continue_probs.argmax(dim=-1)
        else:
            continue_action = continue_dist.sample()
        
        # Generate movement
        direction_raw = self.direction_head(reasoning_state)
        direction = F.normalize(direction_raw, p=2, dim=-1)
        
        step_size = torch.sigmoid(self.step_size_head(reasoning_state)) * 2.0
        
        # Value estimation
        value = self.value_head(reasoning_state)
        
        # Update memory
        if self.training:
            self.update_memory(reasoning_state[0], value[0])
        
        # Move in reasoning space
        next_position = reasoning_state + step_size * direction
        
        # Generate latent thought
        latent_thought = self.thought_projection(next_position)
        
        # Prepare output
        result = {
            'latent_thought': latent_thought.squeeze(0) if single_input else latent_thought,
            'stop': continue_action == 1,
            'position': next_position.squeeze(0) if single_input else next_position,
            'reasoning_state': reasoning_state.squeeze(0) if single_input else reasoning_state
        }
        
        if return_policy_info:
            result.update({
                'action': continue_action.squeeze(0) if single_input else continue_action,
                'log_prob': continue_dist.log_prob(continue_action).squeeze(0) if single_input else continue_dist.log_prob(continue_action),
                'value': value.squeeze() if single_input else value.squeeze(-1),
                'entropy': continue_dist.entropy().squeeze(0) if single_input else continue_dist.entropy(),
                'continue_probs': continue_probs.squeeze(0) if single_input else continue_probs
            })
        
        return result

# ============================================================
# HIERARCHICAL PLANNING MODULE (UNCHANGED)
# ============================================================

class HierarchicalPlanningModule(nn.Module):
    """Hierarchical planning with multiple abstraction levels"""
    
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int,
        num_levels: int = 3,
        recurrent_hidden_size: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.num_levels = num_levels
        self.recurrent_hidden_size = recurrent_hidden_size
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, reasoning_dim),
            nn.LayerNorm(reasoning_dim)
        )
        
        # Recurrent processing
        self.recurrent_cell = nn.GRUCell(reasoning_dim, recurrent_hidden_size)
        
        # Level embeddings
        self.level_embeddings = nn.Embedding(num_levels, reasoning_dim)
        
        # Level selector
        self.level_selector = nn.Sequential(
            nn.Linear(recurrent_hidden_size, recurrent_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(recurrent_hidden_size // 2, num_levels)
        )
        
        # Sub-goal generators for each level
        self.sub_goal_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(recurrent_hidden_size + reasoning_dim, recurrent_hidden_size),
                nn.LayerNorm(recurrent_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(recurrent_hidden_size, reasoning_dim),
                nn.LayerNorm(reasoning_dim)
            ) for _ in range(num_levels)
        ])
        
        # Value estimation for planning
        self.planning_value_head = nn.Sequential(
            nn.Linear(recurrent_hidden_size, recurrent_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(recurrent_hidden_size // 2, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(recurrent_hidden_size, recurrent_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(recurrent_hidden_size // 4, 1)
        )
        
        # Goal refinement
        self.goal_refiner = nn.Sequential(
            nn.Linear(reasoning_dim * 2, reasoning_dim * 2),
            nn.LayerNorm(reasoning_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(reasoning_dim * 2, reasoning_dim),
            nn.LayerNorm(reasoning_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def select_level(self, hidden_state: torch.Tensor, deterministic: bool = False):
        """Select planning level"""
        level_logits = self.level_selector(hidden_state)
        level_probs = F.softmax(level_logits, dim=-1)
        
        if deterministic:
            selected_level = level_probs.argmax(dim=-1)
        else:
            level_dist = torch.distributions.Categorical(level_probs)
            selected_level = level_dist.sample()
        
        return selected_level, level_probs
    
    def generate_sub_goal(self, hidden_state: torch.Tensor, level: int):
        """Generate sub-goal for specific level"""
        level_embedding = self.level_embeddings(torch.tensor(level, device=hidden_state.device))
        
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        if level_embedding.dim() == 1:
            level_embedding = level_embedding.unsqueeze(0)
        
        combined = torch.cat([hidden_state, level_embedding], dim=-1)
        sub_goal = self.sub_goal_generators[level](combined)
        
        return sub_goal
    
    def plan(
        self,
        initial_state: torch.Tensor,
        navigator: MemoryEfficientNavigator,
        max_depth: int = 5,
        uncertainty_threshold: float = 0.1,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """Generate hierarchical plan"""
        # Handle batch dimension
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        
        device = initial_state.device
        
        # Project initial state
        if initial_state.shape[-1] != self.reasoning_dim:
            current_state = self.input_projection(initial_state)
        else:
            current_state = initial_state
        
        # Initialize recurrent state
        hidden = torch.zeros(1, self.recurrent_hidden_size, device=device)
        
        # Storage
        sub_goals = []
        selected_levels = []
        planning_values = []
        uncertainties = []
        refined_positions = []
        
        for depth in range(max_depth):
            # Update recurrent state
            hidden = self.recurrent_cell(current_state, hidden)
            
            # Select planning level
            level, level_probs = self.select_level(hidden, deterministic)
            selected_levels.append(level)
            
            # Generate sub-goal
            level_idx = level.item() if hasattr(level, 'item') else level
            sub_goal = self.generate_sub_goal(hidden, min(level_idx, self.num_levels - 1))
            sub_goals.append(sub_goal)
            
            # Use navigator for refinement
            nav_output = navigator.navigate(sub_goal, deterministic=deterministic)
            refined_position = nav_output['position']
            refined_positions.append(refined_position)
            
            # Refine goal
            combined_goal = torch.cat([sub_goal, refined_position], dim=-1)
            refined_goal = self.goal_refiner(combined_goal)
            
            # Update state
            current_state = 0.7 * current_state + 0.3 * refined_goal
            
            # Compute values
            planning_value = self.planning_value_head(hidden)
            planning_values.append(planning_value)
            
            uncertainty = torch.sigmoid(self.uncertainty_head(hidden))
            uncertainties.append(uncertainty)
            
            # Check stopping
            should_stop = nav_output.get('stop', False)
            if isinstance(should_stop, torch.Tensor):
                should_stop = should_stop.any().item()
            
            if uncertainty.item() < uncertainty_threshold or should_stop:
                break
        
        return {
            'sub_goals': sub_goals,
            'selected_levels': torch.stack(selected_levels) if selected_levels else torch.tensor([]),
            'planning_values': torch.stack(planning_values) if planning_values else torch.tensor([]),
            'uncertainties': torch.stack(uncertainties) if uncertainties else torch.tensor([]),
            'refined_positions': refined_positions,
            'depth_reached': depth + 1,
            'final_state': current_state.squeeze(0)
        }

# ============================================================
# FIXED INTEGRATED VALUE NETWORK
# ============================================================

class IntegratedValueNetwork(nn.Module):
    """FIXED: Unified value network combining navigator and planner information"""
    
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int,
        use_hierarchical: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.use_hierarchical = use_hierarchical
        
        # Hidden state processing
        self.hidden_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 512),
            nn.LayerNorm(512)
        )
        
        # Reasoning state processing
        self.reasoning_projection = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim * 2),
            nn.LayerNorm(reasoning_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(reasoning_dim * 2, 512),
            nn.LayerNorm(512)
        )
        
        # FIXED: Dynamic input size based on what's provided
        # We'll create the value head dynamically based on input
        self.value_head_single = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.value_head_combined = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        reasoning_states: Optional[torch.Tensor] = None,
        navigator_values: Optional[torch.Tensor] = None,
        planner_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute integrated value estimation"""
        features = []
        
        # Process hidden states
        if hidden_states is not None:
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.mean(dim=1)
            elif hidden_states.dim() == 1:
                hidden_states = hidden_states.unsqueeze(0)
            
            hidden_features = self.hidden_projection(hidden_states)
            features.append(hidden_features)
        
        # Process reasoning states
        if reasoning_states is not None:
            if reasoning_states.dim() == 1:
                reasoning_states = reasoning_states.unsqueeze(0)
            
            reasoning_features = self.reasoning_projection(reasoning_states)
            features.append(reasoning_features)
        
        # Combine features and compute value
        if features:
            if len(features) == 1:
                # Single feature source - use single head
                value = self.value_head_single(features[0]).squeeze(-1)
            else:
                # Multiple feature sources - concatenate and use combined head
                combined = torch.cat(features, dim=-1)
                value = self.value_head_combined(combined).squeeze(-1)
        else:
            # Fallback to direct values
            values = []
            if navigator_values is not None:
                values.append(navigator_values)
            if planner_values is not None:
                values.append(planner_values)
            
            if values:
                value = torch.stack(values).mean(dim=0)
            else:
                raise ValueError("No input provided to value network")
        
        # Optional blending with direct values
        if navigator_values is not None or planner_values is not None:
            direct_values = []
            if navigator_values is not None:
                direct_values.append(navigator_values)
            if planner_values is not None:
                direct_values.append(planner_values)
            
            if direct_values:
                direct_value = torch.stack(direct_values).mean(dim=0)
                value = 0.8 * value + 0.2 * direct_value
        
        return value

# ============================================================
# MAIN COCONUT PPO MODEL WITH FIXED LABEL HANDLING
# ============================================================

class CoconutPPO(nn.Module):
    """FIXED: COCONUT with PPO, Memory-Efficient Navigator, and Hierarchical Planning"""
    
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
        use_hierarchical_planning: bool = True,
        num_hierarchy_levels: int = 3,
        recurrent_hidden_size: int = 512,
        memory_size: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.reasoning_dim = reasoning_dim
        self.max_latent_steps = max_latent_steps
        self.use_hierarchical_planning = use_hierarchical_planning
        
        # Token IDs
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        
        # Memory-efficient navigator
        self.navigator = MemoryEfficientNavigator(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
            memory_size=memory_size,
            dropout_rate=dropout_rate
        )
        
        # Hierarchical planner
        if use_hierarchical_planning:
            self.planner = HierarchicalPlanningModule(
                hidden_size=hidden_size,
                reasoning_dim=reasoning_dim,
                num_levels=num_hierarchy_levels,
                recurrent_hidden_size=recurrent_hidden_size,
                dropout_rate=dropout_rate
            )
        else:
            self.planner = None
        
        # Integrated value network
        self.value_network = IntegratedValueNetwork(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
            use_hierarchical=use_hierarchical_planning,
            dropout_rate=dropout_rate
        )
        
        # Move to same device as base model
        device = next(base_model.parameters()).device
        self.navigator = self.navigator.to(device)
        if self.planner is not None:
            self.planner = self.planner.to(device)
        self.value_network = self.value_network.to(device)
        
        # Enable gradient checkpointing if available
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
        
        logger.info(f"Initialized CoconutPPO with reasoning_dim={reasoning_dim}, memory_size={memory_size}")
    
    def compute_value(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        trajectory: Optional[Dict] = None
    ) -> torch.Tensor:
        """Compute integrated value estimation"""
        navigator_values = None
        planner_values = None
        reasoning_states = None
        
        if trajectory:
            # Extract navigator values
            if 'values' in trajectory and trajectory['values']:
                if isinstance(trajectory['values'], list):
                    navigator_values = torch.stack(trajectory['values']).mean()
                else:
                    navigator_values = trajectory['values']
            
            # Extract reasoning states
            if 'reasoning_states' in trajectory and trajectory['reasoning_states']:
                if isinstance(trajectory['reasoning_states'], list):
                    reasoning_states = trajectory['reasoning_states'][-1]
                else:
                    reasoning_states = trajectory['reasoning_states']
            
            # Extract planner values
            if 'planning_info' in trajectory and trajectory['planning_info']:
                plan_info = trajectory['planning_info']
                if 'planning_values' in plan_info and plan_info['planning_values'].numel() > 0:
                    planner_values = plan_info['planning_values'].mean()
        
        return self.value_network(
            hidden_states=hidden_states,
            reasoning_states=reasoning_states,
            navigator_values=navigator_values,
            planner_values=planner_values
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_trajectory: bool = True,
        use_planning: Optional[bool] = None,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with navigation and planning"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Determine planning usage
        use_planning = use_planning if use_planning is not None else self.use_hierarchical_planning
        use_planning = use_planning and self.planner is not None
        
        # Get initial embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Process batch
        batch_trajectories = []
        batch_latent_sequences = []
        
        for b in range(batch_size):
            trajectory = {
                'states': [],
                'reasoning_states': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'latent_embeds': [],
                'planning_info': None
            }
            
            # Get initial hidden state
            with torch.no_grad():
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[b:b+1],
                    attention_mask=attention_mask[b:b+1] if attention_mask is not None else None,
                    output_hidden_states=True,
                    use_cache=False
                )
            
            current_state = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
            
            # Use hierarchical planning
            planning_output = None
            if use_planning:
                planning_output = self.planner.plan(
                    current_state,
                    self.navigator,
                    max_depth=min(5, self.max_latent_steps),
                    deterministic=deterministic
                )
                trajectory['planning_info'] = planning_output
            
            # Navigate through reasoning space
            latent_sequence = []
            for step in range(self.max_latent_steps):
                # Use planning guidance if available
                if planning_output and step < len(planning_output['sub_goals']):
                    sub_goal = planning_output['sub_goals'][step]
                    
                    # Blend with current state
                    if current_state.shape[-1] != self.reasoning_dim:
                        current_state_proj = self.navigator.state_projection(current_state.unsqueeze(0)).squeeze(0)
                    else:
                        current_state_proj = current_state
                    
                    # Weighted combination
                    if step < len(planning_output['uncertainties']):
                        uncertainty = planning_output['uncertainties'][step].mean().item()
                        blend_weight = 1.0 - uncertainty
                    else:
                        blend_weight = 0.5
                    
                    nav_input = (1 - blend_weight) * current_state_proj + blend_weight * sub_goal.squeeze()
                else:
                    nav_input = current_state
                
                # Navigate
                nav_output = self.navigator.navigate(
                    nav_input,
                    deterministic=deterministic,
                    return_policy_info=True
                )
                
                # Store trajectory
                if return_trajectory:
                    trajectory['states'].append(current_state.detach())
                    trajectory['reasoning_states'].append(nav_output.get('reasoning_state', current_state).detach())
                    trajectory['actions'].append(nav_output['action'])
                    trajectory['log_probs'].append(nav_output['log_prob'])
                    trajectory['values'].append(nav_output['value'])
                
                # Check stopping
                should_stop = nav_output.get('stop', False)
                if isinstance(should_stop, torch.Tensor):
                    should_stop = should_stop.item() if should_stop.numel() == 1 else should_stop.any().item()
                
                if should_stop:
                    break
                
                # Store latent thought
                latent_thought = nav_output['latent_thought']
                latent_sequence.append(latent_thought)
                
                if return_trajectory:
                    trajectory['latent_embeds'].append(latent_thought.detach())
                
                # Update state
                current_state = latent_thought
            
            # Compute integrated value
            if return_trajectory:
                integrated_value = self.compute_value(
                    hidden_states=current_state.unsqueeze(0),
                    trajectory=trajectory
                )
                trajectory['integrated_value'] = integrated_value
            
            batch_trajectories.append(trajectory)
            batch_latent_sequences.append(latent_sequence)
        
        # Process with latent thoughts
        max_latent_len = max(len(seq) for seq in batch_latent_sequences) if batch_latent_sequences else 0
        
        if max_latent_len > 0:
            # FIXED: Properly adjust labels to match enhanced embeddings size
            if labels is not None:
                # Get original sequence length
                original_seq_len = inputs_embeds.shape[1]
                
                # New sequence length after adding latent tokens
                new_seq_len = original_seq_len + max_latent_len
                
                # Create new labels tensor
                new_labels = torch.full(
                    (batch_size, new_seq_len),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                
                # Copy original labels, shifting them to account for latent tokens
                # First token stays in place
                new_labels[:, 0] = labels[:, 0] if labels.shape[1] > 0 else -100
                
                # Rest of the labels go after latent tokens
                if labels.shape[1] > 1:
                    # Only copy as many labels as we have space for
                    copy_len = min(labels.shape[1] - 1, new_seq_len - max_latent_len - 1)
                    new_labels[:, max_latent_len + 1:max_latent_len + 1 + copy_len] = labels[:, 1:1 + copy_len]
                
                labels = new_labels
            
            # Pad latent sequences
            padded_latent_sequences = []
            for seq in batch_latent_sequences:
                if len(seq) == 0:
                    padded_seq = [torch.zeros(self.hidden_size, device=device) for _ in range(max_latent_len)]
                else:
                    padding = [torch.zeros_like(seq[0]) for _ in range(max_latent_len - len(seq))]
                    padded_seq = seq + padding
                
                padded_latent_sequences.append(torch.stack(padded_seq))
            
            latent_embeds = torch.stack(padded_latent_sequences)
            
            # Combine embeddings
            enhanced_embeds = torch.cat([
                inputs_embeds[:, :1, :],
                latent_embeds,
                inputs_embeds[:, 1:, :]
            ], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                latent_attention = torch.ones(batch_size, max_latent_len, dtype=attention_mask.dtype, device=device)
                enhanced_mask = torch.cat([
                    attention_mask[:, :1],
                    latent_attention,
                    attention_mask[:, 1:]
                ], dim=1)
            else:
                enhanced_mask = None
            
            # Ensure labels match embeddings shape
            if labels is not None and labels.shape[1] != enhanced_embeds.shape[1]:
                # Truncate or pad labels to match
                if labels.shape[1] > enhanced_embeds.shape[1]:
                    labels = labels[:, :enhanced_embeds.shape[1]]
                else:
                    padding = torch.full(
                        (batch_size, enhanced_embeds.shape[1] - labels.shape[1]),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    labels = torch.cat([labels, padding], dim=1)
            
            # Forward through base model
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
            result['trajectory'] = batch_trajectories[0] if batch_size == 1 else batch_trajectories
        
        return result
