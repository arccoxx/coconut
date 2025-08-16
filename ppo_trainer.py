"""
PPO Trainer optimized for Llama 3-8B on A100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class PPOTrainer:
    """PPO trainer with memory optimizations for A100"""
    
    def __init__(
        self,
        model: CoconutPPO,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # PPO hyperparameters
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        
        # Optimizer with different LRs
        self.optimizer = torch.optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': 1e-5},  # Lower LR for base
            {'params': model.navigator.parameters(), 'lr': 3e-4},   # Higher LR for navigator
            {'params': model.policy_network.parameters(), 'lr': 3e-4},
            {'params': model.value_network.parameters(), 'lr': 3e-4}
        ], weight_decay=0.01)
        
        # Rollout buffer (memory efficient)
        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.get('rollout_buffer_size', 512),
            device=device
        )
        
        # Mixed precision training for A100
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics tracking
        self.metrics = deque(maxlen=100)
        
    def train_step(self, batch: Dict) -> Dict:
        """Single training step with PPO"""
        self.model.train()
        
        # Collect trajectories with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                labels=batch.get('labels', None),
                return_trajectory=True
            )
        
        # Compute rewards
        rewards = self.compute_rewards(outputs, batch)
        
        # Store in rollout buffer
        trajectory = outputs['trajectory']
        for i in range(len(trajectory['states'])):
            self.rollout_buffer.add(
                state=trajectory['states'][i],
                action=trajectory['actions'][i],
                log_prob=trajectory['log_probs'][i],
                value=trajectory['values'][i],
                reward=rewards[i] if i < len(rewards) else 0
            )
        
        # PPO update when buffer is full
        ppo_loss = 0
        if self.rollout_buffer.is_ready():
            ppo_loss = self.ppo_update()
            self.rollout_buffer.clear()
        
        # Task loss
        task_loss = outputs.get('loss', torch.tensor(0.0))
        
        metrics = {
            'task_loss': task_loss.item() if task_loss is not None else 0,
            'ppo_loss': ppo_loss,
            'trajectory_len': len(trajectory['states']),
            'reward': rewards.mean().item() if len(rewards) > 0 else 0
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def compute_rewards(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute rewards for trajectory"""
        rewards = []
        
        # Task reward (based on loss improvement)
        if outputs.get('loss') is not None:
            task_reward = -outputs['loss'].item()  # Negative loss as reward
        else:
            task_reward = 0
        
        trajectory = outputs['trajectory']
        num_steps = len(trajectory['states'])
        
        for i in range(num_steps):
            if i == num_steps - 1:
                # Final step gets task reward
                reward = task_reward
            else:
                # Intermediate steps get small negative reward (efficiency)
                reward = -0.01
            
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def ppo_update(self) -> float:
        """PPO parameter update"""
        # Get data from buffer
        states, actions, old_log_probs, values, rewards = self.rollout_buffer.get()
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Mini-batch updates
            for batch_indices in self.get_minibatches(len(states), 32):
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                with torch.cuda.amp.autocast():
                    # Recompute from states
                    nav_outputs = self.model.navigator.navigate(
                        batch_states,
                        return_policy_info=True
                    )
                    
                    log_probs = nav_outputs['log_prob']
                    entropy = nav_outputs['entropy']
                    values_pred = nav_outputs['value']
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Backward with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
        
        return total_loss / (self.ppo_epochs * len(states) // 32)
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """GAE computation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        
        return advantages
    
    def get_minibatches(self, size: int, batch_size: int):
        """Generate minibatch indices"""
        indices = np.arange(size)
        np.random.shuffle(indices)
        for start in range(0, size, batch_size):
            yield indices[start:start + batch_size]

class RolloutBuffer:
    """Memory-efficient rollout buffer"""
    
    def __init__(self, buffer_size: int, device: str):
        self.buffer_size = buffer_size
        self.device = device
        self.reset()
        
    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.ptr = 0
        
    def add(self, state, action, log_prob, value, reward):
        if self.ptr < self.buffer_size:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.ptr += 1
            
    def get(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        rewards = torch.tensor(self.rewards, device=self.device)
        return states, actions, log_probs, values, rewards
    
    def is_ready(self):
        return self.ptr >= self.buffer_size
    
    def clear(self):
        self.reset()
