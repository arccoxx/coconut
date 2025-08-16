"""Full training script for COCONUT PPO with Llama 3"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_ppo import CoconutPPO
from prepare_dataset import create_dataloaders
import logging
from tqdm import tqdm
import wandb
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOTrainer:
    """Full PPO trainer with proper advantages and batching"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Separate optimizers for different learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': config['lr_base']},
            {'params': model.navigator.parameters(), 'lr': config['lr_navigator']}
        ], weight_decay=0.01)
        
        # PPO parameters
        self.clip_epsilon = config['clip_epsilon']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.ppo_epochs = config['ppo_epochs']
        
        # Rollout buffer
        self.rollout_buffer = []
        self.max_buffer_size = config['rollout_buffer_size']
        
        # Metrics
        self.metrics = deque(maxlen=100)
        
    def compute_advantages(self, rewards, values, dones):
        """Compute GAE advantages"""
        advantages = []
        returns = []
        
        advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_rewards(self, outputs, batch):
        """Compute rewards based on loss improvement"""
        rewards = []
        
        # Base reward from task loss
        if outputs.get('loss') is not None:
            base_reward = -outputs['loss'].item()  # Negative loss as reward
        else:
            base_reward = 0
        
        # Trajectory length penalty
        traj_len = len(outputs['trajectory']['states'])
        length_penalty = -0.01 * traj_len
        
        # Create reward for each step
        for i in range(traj_len):
            if i == traj_len - 1:
                # Final step gets the task reward
                rewards.append(base_reward + length_penalty)
            else:
                # Intermediate steps get small negative reward
                rewards.append(-0.01)
        
        return rewards
    
    def ppo_update(self):
        """Perform PPO update on collected rollouts"""
        if len(self.rollout_buffer) == 0:
            return 0
        
        # Aggregate all rollouts
        all_states = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        
        for rollout in self.rollout_buffer:
            all_states.extend(rollout['states'])
            all_actions.extend(rollout['actions'])
            all_log_probs.extend(rollout['log_probs'])
            all_advantages.extend(rollout['advantages'])
            all_returns.extend(rollout['returns'])
        
        # Convert to tensors
        states = torch.stack(all_states)
        actions = torch.stack(all_actions)
        old_log_probs = torch.stack(all_log_probs)
        advantages = torch.stack(all_advantages)
        returns = torch.stack(all_returns)
        
        total_loss = 0
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Mini-batch training
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), 32):
                end = min(start + 32, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    nav_outputs = self.model.navigator.navigate(
                        batch_states,
                        return_policy_info=True
                    )
                
                # Compute losses
                log_probs = nav_outputs['log_prob']
                values = nav_outputs['value']
                entropy = nav_outputs['entropy']
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Clear buffer
        self.rollout_buffer = []
        
        return total_loss / (self.ppo_epochs * len(states) // 32)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'task_loss': [],
            'ppo_loss': [],
            'rewards': [],
            'trajectory_lengths': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward pass with trajectory
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    return_trajectory=True
                )
            
            # Compute rewards
            rewards = self.compute_rewards(outputs, batch)
            
            # Get trajectory
            trajectory = outputs['trajectory']
            if len(trajectory['states']) > 0:
                # Compute advantages
                values = [v.item() for v in trajectory['values']]
                dones = [0] * (len(values) - 1) + [1]  # Last step is done
                
                advantages, returns = self.compute_advantages(rewards, values, dones)
                
                # Store rollout
                self.rollout_buffer.append({
                    'states': trajectory['states'],
                    'actions': trajectory['actions'],
                    'log_probs': trajectory['log_probs'],
                    'advantages': advantages,
                    'returns': returns
                })
            
            # Backward pass on task loss
            if outputs['loss'] is not None:
                task_loss = outputs['loss']
                self.optimizer.zero_grad()
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_metrics['task_loss'].append(task_loss.item())
            
            # PPO update when buffer is full
            if len(self.rollout_buffer) >= 10:  # Update every 10 rollouts
                ppo_loss = self.ppo_update()
                epoch_metrics['ppo_loss'].append(ppo_loss)
            
            # Track metrics
            epoch_metrics['rewards'].append(np.mean(rewards))
            epoch_metrics['trajectory_lengths'].append(len(trajectory['states']))
            
            # Update progress bar
            progress_bar.set_postfix({
                'task_loss': np.mean(epoch_metrics['task_loss'][-10:]) if epoch_metrics['task_loss'] else 0,
                'reward': np.mean(epoch_metrics['rewards'][-10:]) if epoch_metrics['rewards'] else 0,
                'traj_len': np.mean(epoch_metrics['trajectory_lengths'][-10:]) if epoch_metrics['trajectory_lengths'] else 0
            })
            
            # Log to wandb
            if batch_idx % 10 == 0 and wandb.run:
                wandb.log({
                    'task_loss': epoch_metrics['task_loss'][-1] if epoch_metrics['task_loss'] else 0,
                    'ppo_loss': epoch_metrics['ppo_loss'][-1] if epoch_metrics['ppo_loss'] else 0,
                    'reward': epoch_metrics['rewards'][-1],
                    'trajectory_length': epoch_metrics['trajectory_lengths'][-1],
                    'epoch': epoch,
                    'step': epoch * len(train_loader) + batch_idx
                })
        
        return epoch_metrics

def main():
    # Configuration
    config = {
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'batch_size': 2,
        'lr_base': 1e-5,
        'lr_navigator': 3e-4,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 4,
        'rollout_buffer_size': 100,
        'num_epochs': 5
    }
    
    # Initialize wandb
    wandb.init(project="coconut-llama3-ppo", config=config)
    
    # Load model and tokenizer
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Create COCONUT model
    model = CoconutPPO(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=256,
        max_latent_steps=4
    )
    
    # Move navigator to CUDA
    model.navigator = model.navigator.cuda()
    
    logger.info(f"Model ready with {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(tokenizer, batch_size=config['batch_size'])
    
    # Create trainer
    trainer = PPOTrainer(model, config)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config['num_epochs']):
        # Train
        epoch_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Log epoch summary
        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  Avg task loss: {np.mean(epoch_metrics['task_loss']):.4f}")
        logger.info(f"  Avg reward: {np.mean(epoch_metrics['rewards']):.4f}")
        logger.info(f"  Avg trajectory length: {np.mean(epoch_metrics['trajectory_lengths']):.2f}")
        
        # Save checkpoint
        if epoch % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': epoch_metrics
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
            logger.info(f"Checkpoint saved")
    
    logger.info("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
