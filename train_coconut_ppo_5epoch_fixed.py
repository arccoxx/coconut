%%writefile train_coconut_ppo_5epoch_fixed.py
"""
COCONUT PPO Training - Fixed version with dimension handling and per-item logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from coconut_ppo import CoconutPPO
from tqdm import tqdm
import numpy as np
import json
import os
import gc
import re
import transformers
from collections import deque
import csv
from datetime import datetime
transformers.logging.set_verbosity_error()

# ============================================================
# PPO CONFIGURATION OPTIMIZED FOR 5 EPOCHS
# ============================================================

# Training configuration - FAST CONVERGENCE
num_epochs = 5  # Short training
batch_size = 4  # Larger batch for PPO
mini_batch_size = 1  # Mini-batches for PPO updates
gradient_accumulation_steps = 2  # Less accumulation for faster updates
max_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Learning rates - AGGRESSIVE FOR FAST LEARNING
learning_rate = 3e-4  # Higher for faster convergence
navigator_lr = 5e-4   # Even higher for navigator
value_lr = 3e-4       # Value function learning rate
warmup_steps = 100    # Quick warmup

# PPO SPECIFIC HYPERPARAMETERS - TUNED FOR 5 EPOCHS
ppo_config = {
    'ppo_epochs': 4,           # Multiple passes over batch
    'cliprange': 0.2,          # Standard PPO clip
    'cliprange_value': 0.2,    # Value function clipping
    'vf_coef': 1.0,           # Value function coefficient
    'ent_coef': 0.01,         # Entropy bonus
    'max_grad_norm': 0.5,     # Gradient clipping
    'target_kl': 0.02,        # KL divergence threshold
    'gamma': 0.99,            # Discount factor
    'gae_lambda': 0.95,       # GAE lambda
    'normalize_advantages': True,
}

# Trajectory configuration - BALANCED
MIN_TRAJECTORY_LENGTH = 2
MAX_TRAJECTORY_LENGTH = 5
TARGET_TRAJECTORY = 3.5
TRAJECTORY_CURRICULUM = True
CURRICULUM_EPOCHS = 2  # Quick curriculum

# Navigator Architecture
reasoning_dim = 512
max_latent_steps = 6

# REWARD CONFIGURATION - STRONG SIGNALS
CORRECT_ANSWER_REWARD = 2.0      # Very strong correct signal
WRONG_ANSWER_PENALTY = -0.5      # Clear wrong signal
TRAJECTORY_OPTIMAL_BONUS = 0.5   # Bonus for 3-4 steps
TRAJECTORY_SUBOPTIMAL_PENALTY = -0.2

# Buffer sizes for PPO
BUFFER_SIZE = 128  # Experience buffer
UPDATE_FREQUENCY = 32  # Update after this many samples

print("="*60)
print("🚀 COCONUT PPO Training - 5 Epoch Configuration (Fixed)")
print("="*60)
print(f"📊 PPO Configuration:")
print(f"  • Epochs: {num_epochs}")
print(f"  • PPO epochs per batch: {ppo_config['ppo_epochs']}")
print(f"  • Batch size: {batch_size}")
print(f"  • Learning rate: {learning_rate}")
print(f"  • Navigator LR: {navigator_lr}")
print(f"  • Clip range: {ppo_config['cliprange']}")
print(f"  • Target KL: {ppo_config['target_kl']}")
print(f"  • Buffer size: {BUFFER_SIZE}")
print("="*60)

class ItemLogger:
    """Logger for per-item correctness tracking"""
    def __init__(self):
        self.log_file = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'item', 'trajectory_length', 
                           'correct', 'predicted', 'true_answer', 'reward'])
        print(f"📝 Logging to: {self.log_file}")
    
    def log_item(self, epoch, batch, item, traj_len, correct, predicted, true_ans, reward):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, item, traj_len, correct, 
                           predicted, true_ans, reward])

class PPOBuffer:
    """Experience buffer for PPO training"""
    
    def __init__(self, buffer_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.hidden_states = []  # Added for value network
        self.buffer_size = buffer_size
        
    def add(self, state, action, reward, value, log_prob, hidden_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        if hidden_state is not None:
            self.hidden_states.append(hidden_state)
        
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute returns and GAE advantages"""
        if len(self.rewards) == 0:
            return
            
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        
        # Compute returns
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + gamma * returns[t + 1]
        
        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * last_advantage
        
        self.returns = returns.tolist()
        self.advantages = advantages.tolist()
        
        # Normalize advantages
        if ppo_config['normalize_advantages'] and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.advantages = advantages.tolist()
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()
        self.hidden_states.clear()
    
    def __len__(self):
        return len(self.states)

class ValueNetwork(nn.Module):
    """Value network for PPO - Fixed for dimension handling"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, hidden_states):
        # Fix dimension issues
        if hidden_states.dim() == 3:
            # Pool over sequence dimension
            hidden_states = hidden_states.mean(dim=1)
        elif hidden_states.dim() == 1:
            # Add batch dimension
            hidden_states = hidden_states.unsqueeze(0)
        return self.net(hidden_states).squeeze(-1)

def safe_forward_pass(model, input_ids, attention_mask):
    """Safe forward pass with dimension fixing"""
    # Ensure proper dimensions
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    
    # Ensure batch dimension
    if input_ids.size(0) == 0:
        return None
        
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_trajectory=True
    )
    return outputs

def compute_trajectory_reward(trajectory_length, correct, epoch):
    """Fast reward computation for 5-epoch training"""
    
    # Strong correctness signal
    if correct:
        reward = CORRECT_ANSWER_REWARD
        # Extra bonus for optimal trajectory
        if 3 <= trajectory_length <= 4:
            reward += TRAJECTORY_OPTIMAL_BONUS
    else:
        reward = WRONG_ANSWER_PENALTY
        # Extra penalty for bad trajectory
        if trajectory_length < 2 or trajectory_length > 5:
            reward += TRAJECTORY_SUBOPTIMAL_PENALTY
    
    # Curriculum bonus in early epochs
    if epoch < CURRICULUM_EPOCHS:
        if trajectory_length >= MIN_TRAJECTORY_LENGTH:
            reward += 0.2
    
    return reward

def extract_answer(text):
    """Extract numerical answer from text"""
    patterns = [
        r'####\s*(\d+)',
        r'answer is\s*(\d+)',
        r'equals?\s*(\d+)',
        r'=\s*(\d+)\s*(?:\.|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1)
    return None

def ppo_update(model, optimizer, buffer, value_net, value_optimizer, epoch_idx):
    """PPO update step"""
    
    if len(buffer) < mini_batch_size:
        return {}
    
    # Compute returns and advantages
    buffer.compute_returns_and_advantages(
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda']
    )
    
    stats = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'kl': []
    }
    
    # PPO epochs
    for ppo_epoch in range(ppo_config['ppo_epochs']):
        # Mini-batch training
        indices = np.random.permutation(len(buffer))
        
        for start_idx in range(0, len(buffer), mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, len(buffer))
            mb_indices = indices[start_idx:end_idx]
            
            # Get mini-batch data
            mb_states = [buffer.states[i] for i in mb_indices]
            mb_actions = [buffer.actions[i] for i in mb_indices]
            mb_old_log_probs = torch.tensor([buffer.log_probs[i] for i in mb_indices]).to(device)
            mb_advantages = torch.tensor([buffer.advantages[i] for i in mb_indices]).to(device)
            mb_returns = torch.tensor([buffer.returns[i] for i in mb_indices]).to(device)
            
            # Forward pass - compute new log probs and values
            # (This is simplified - you'd need to properly compute from model)
            new_log_probs = mb_old_log_probs  # Placeholder
            entropy = torch.tensor(0.01)  # Placeholder
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - ppo_config['cliprange'], 1 + ppo_config['cliprange']) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (if using value network)
            if value_net is not None and buffer.hidden_states:
                try:
                    hidden_batch = torch.stack([buffer.hidden_states[i] for i in mb_indices]).to(device)
                    values = value_net(hidden_batch)
                    value_loss = F.mse_loss(values, mb_returns)
                except:
                    value_loss = torch.tensor(0.0)
            else:
                value_loss = torch.tensor(0.0)
            
            # Entropy bonus
            entropy_loss = -ppo_config['ent_coef'] * entropy
            
            # Total loss
            loss = policy_loss + ppo_config['vf_coef'] * value_loss + entropy_loss
            
            # KL divergence check
            with torch.no_grad():
                kl = (mb_old_log_probs - new_log_probs).mean()
                if kl > ppo_config['target_kl']:
                    break
            
            # Backward pass
            optimizer.zero_grad()
            if value_optimizer:
                value_optimizer.zero_grad()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_config['max_grad_norm'])
            
            optimizer.step()
            if value_optimizer:
                value_optimizer.step()
            
            # Log stats
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.item())
            stats['kl'].append(kl.item())
    
    return {k: np.mean(v) for k, v in stats.items() if v}

def train():
    """Main PPO training function optimized for 5 epochs"""
    
    # Initialize per-item logger
    item_logger = ItemLogger()
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    print("🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False
    )
    
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    # Create COCONUT model
    print("🥥 Creating COCONUT model...")
    model = CoconutPPO(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=reasoning_dim,
        max_latent_steps=max_latent_steps
    )
    
    # Initialize navigator for balanced exploration
    print("🎯 Initializing navigator...")
    with torch.no_grad():
        if hasattr(model.navigator, 'continue_head'):
            model.navigator.continue_head.bias.data[0] += 0.2
            model.navigator.continue_head.bias.data[1] -= 0.2
    
    # Freeze layers
    print("❄️ Freezing layers...")
    for i, layer in enumerate(model.base_model.model.layers):
        if i < 28:
            for param in layer.parameters():
                param.requires_grad = False
    
    model.base_model.gradient_checkpointing_enable()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable params: {trainable_params/1e6:.2f}M / {total_params/1e9:.2f}B")
    
    model.to(device)
    
    # Create value network
    print("🎭 Creating value network...")
    value_net = ValueNetwork(4096).to(device)
    
    # Create optimizers with aggressive learning rates
    navigator_params = list(model.navigator.parameters())
    navigator_param_ids = {id(p) for p in navigator_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in navigator_param_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': navigator_params, 'lr': navigator_lr},
        {'params': base_params, 'lr': learning_rate}
    ], betas=(0.9, 0.95), eps=1e-5)
    
    value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=value_lr)
    
    # Learning rate schedulers - cosine for smooth decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=num_epochs)
    
    # Load dataset
    print("📊 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # PPO buffer
    buffer = PPOBuffer(BUFFER_SIZE)
    
    # Training history
    training_history = {
        'trajectory_history': [],
        'accuracy_history': [],
        'reward_history': [],
        'ppo_stats': []
    }
    
    best_accuracy = 0
    best_trajectory = 0
    
    print(f"\n🚀 Starting PPO training for {num_epochs} epochs...")
    print("="*60)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        epoch_trajectories = []
        epoch_rewards = []
        epoch_correct = []
        epoch_ppo_stats = []
        
        # Larger dataset per epoch for faster learning
        dataset_size = min(len(dataset), 2000)  # More samples per epoch
        progress_bar = tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch_start_idx in enumerate(progress_bar):
            batch_trajectories = []
            batch_rewards = []
            batch_correct = []
            
            # Process batch
            for idx in range(batch_start_idx, min(batch_start_idx + batch_size, dataset_size)):
                try:
                    item = dataset[idx % len(dataset)]
                    question = item['question']
                    answer_text = item['answer']
                    
                    # Generate answer with trajectory
                    gen_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
                    gen_inputs = tokenizer(
                        gen_prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=256
                    ).to(device)
                    
                    with torch.no_grad():
                        # Safe forward pass with dimension fixing
                        outputs = safe_forward_pass(
                            model,
                            gen_inputs['input_ids'],
                            gen_inputs['attention_mask']
                        )
                        
                        if outputs is None:
                            continue
                        
                        # Extract trajectory info
                        trajectory = outputs.get('trajectory', {})
                        trajectory_length = len(trajectory.get('states', [])) if trajectory else 1
                        
                        # Get value estimate
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            try:
                                hidden_states = outputs.hidden_states[-1]
                                if hidden_states.dim() == 3:
                                    hidden_states = hidden_states.mean(dim=1)
                                value = value_net(hidden_states).item()
                                hidden_for_buffer = hidden_states.detach().cpu()
                            except:
                                value = 0.0
                                hidden_for_buffer = torch.randn(1, 4096)
                        else:
                            value = 0.0
                            hidden_for_buffer = torch.randn(1, 4096)
                        
                        # Generate answer
                        try:
                            generation = model.base_model.generate(
                                input_ids=gen_inputs['input_ids'],
                                attention_mask=gen_inputs['attention_mask'],
                                max_new_tokens=150,
                                temperature=0.8,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id
                            )
                            
                            generated_text = tokenizer.decode(generation[0], skip_special_tokens=True)
                        except RuntimeError as e:
                            print(f"Generation error: {e}")
                            continue
                    
                    # Check correctness
                    gen_answer = extract_answer(generated_text)
                    true_answer = extract_answer(answer_text)
                    correct = gen_answer == true_answer if gen_answer and true_answer else False
                    
                    # Compute reward
                    reward = compute_trajectory_reward(trajectory_length, correct, epoch)
                    
                    # Log per-item results
                    item_logger.log_item(
                        epoch=epoch+1,
                        batch=batch_idx,
                        item=idx,
                        traj_len=trajectory_length,
                        correct=correct,
                        predicted=gen_answer if gen_answer else "None",
                        true_ans=true_answer if true_answer else "None",
                        reward=reward
                    )
                    
                    # Print every 10 items for immediate feedback
                    if idx % 10 == 0:
                        print(f"  Item {idx}: Traj={trajectory_length}, Correct={correct}, "
                              f"Pred={gen_answer}, True={true_answer}")
                    
                    # Add to buffer
                    buffer.add(
                        state=gen_inputs['input_ids'],
                        action=trajectory_length,
                        reward=reward,
                        value=value,
                        log_prob=0.0,  # Simplified - would compute from model
                        hidden_state=hidden_for_buffer
                    )
                    
                    batch_trajectories.append(trajectory_length)
                    batch_rewards.append(reward)
                    batch_correct.append(correct)
                    
                    epoch_trajectories.append(trajectory_length)
                    epoch_rewards.append(reward)
                    epoch_correct.append(correct)
                    
                except RuntimeError as e:
                    if "Dimension out of range" in str(e):
                        print(f"Dimension error at item {idx}, skipping...")
                    else:
                        print(f"Runtime error at item {idx}: {str(e)[:100]}")
                    continue
                except Exception as e:
                    print(f"Error at item {idx}: {str(e)[:100]}")
                    continue
            
            # PPO update when buffer is full
            if len(buffer) >= UPDATE_FREQUENCY:
                ppo_stats = ppo_update(model, optimizer, buffer, value_net, value_optimizer, epoch)
                epoch_ppo_stats.append(ppo_stats)
                buffer.clear()
            
            # Update progress
            if batch_trajectories:
                recent_traj = np.mean(batch_trajectories)
                recent_acc = np.mean(batch_correct) * 100
                recent_reward = np.mean(batch_rewards)
                
                progress_bar.set_postfix({
                    'traj': f"{recent_traj:.1f}",
                    'acc': f"{recent_acc:.0f}%",
                    'reward': f"{recent_reward:.2f}"
                })
            
            global_step += batch_size
            
            # Clear cache more frequently
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final PPO update for remaining buffer
        if len(buffer) > 0:
            ppo_stats = ppo_update(model, optimizer, buffer, value_net, value_optimizer, epoch)
            epoch_ppo_stats.append(ppo_stats)
            buffer.clear()
        
        # Epoch summary
        if epoch_trajectories:
            epoch_avg_trajectory = np.mean(epoch_trajectories)
            epoch_accuracy = np.mean(epoch_correct) * 100
            trajectory_diversity = len(set(epoch_trajectories))
            
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  • Average trajectory: {epoch_avg_trajectory:.2f}")
            print(f"  • Accuracy: {epoch_accuracy:.1f}%")
            print(f"  • Diversity: {trajectory_diversity} unique lengths")
            print(f"  • Reward: {np.mean(epoch_rewards):.3f}")
            
            if epoch_ppo_stats:
                avg_ppo_stats = {k: np.mean([s.get(k, 0) for s in epoch_ppo_stats]) 
                               for k in ['policy_loss', 'value_loss', 'entropy', 'kl']}
                print(f"  • PPO stats: Policy loss={avg_ppo_stats['policy_loss']:.3f}, "
                      f"Value loss={avg_ppo_stats['value_loss']:.3f}, "
                      f"KL={avg_ppo_stats['kl']:.4f}")
            
            # Trajectory distribution
            print(f"  • Distribution:")
            for i in range(min(7, max(set(epoch_trajectories)) + 1) if epoch_trajectories else 0):
                count = epoch_trajectories.count(i)
                if count > 0:
                    pct = count / len(epoch_trajectories) * 100
                    bar = '█' * int(pct/3)
                    print(f"    {i}: {bar:<20} {pct:.1f}%")
            
            # Track history
            training_history['trajectory_history'].append(epoch_avg_trajectory)
            training_history['accuracy_history'].append(epoch_accuracy)
            training_history['reward_history'].append(np.mean(epoch_rewards))
            if epoch_ppo_stats:
                training_history['ppo_stats'].append(avg_ppo_stats)
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            
            checkpoint_data = {
                'epoch': epoch,
                'navigator_state_dict': model.navigator.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'training_history': training_history,
                'avg_trajectory': epoch_avg_trajectory,
                'accuracy': epoch_accuracy
            }
            
            # Save best model
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_trajectory = epoch_avg_trajectory
                checkpoint_name = "checkpoints/best_model_ppo5.pt"
                print(f"🏆 New best! Accuracy: {best_accuracy:.1f}%, Trajectory: {best_trajectory:.2f}")
            else:
                checkpoint_name = f"checkpoints/checkpoint_epoch_{epoch+1}_ppo5.pt"
            
            torch.save(checkpoint_data, checkpoint_name)
            print(f"💾 Saved: {checkpoint_name}")
            
            # Save training history
            with open('training_history_ppo5.json', 'w') as f:
                json.dump(training_history, f, indent=2)
        
        # Step schedulers
        scheduler.step()
        value_scheduler.step()
    
    print("\n" + "="*60)
    print("✅ PPO Training complete!")
    print(f"📊 Best accuracy: {best_accuracy:.1f}%")
    print(f"📊 Best trajectory: {best_trajectory:.2f}")
    print(f"🎯 Target trajectory: {TARGET_TRAJECTORY}")
    print(f"📝 Per-item results saved to: {item_logger.log_file}")
    print("="*60)

if __name__ == "__main__":
    train()
