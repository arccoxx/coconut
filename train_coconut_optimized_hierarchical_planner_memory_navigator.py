"""
Optimized COCONUT Training Script with Anti-Mode-Collapse Measures and NaN Prevention
Combines best practices with comprehensive numerical stability fixes
"""

import warnings
import logging
import os
import sys
from contextlib import contextmanager, redirect_stderr
import io

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Suppress warnings
@contextmanager
def suppress_output():
    """Suppress both stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        with redirect_stderr(devnull):
            yield

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import transformers
transformers.logging.set_verbosity_error()

logging.getLogger("transformers.models.llama.modeling_llama").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import gc
import re
import csv
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

# Import model
from coconut_memory_navigator_hierarchical_planner_ppo_fixed import CoconutPPO, PPOReplayBuffer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# STABLE TRAINING CONFIGURATION
# ============================================================

class TrainingConfig:
    """Optimized configuration with enhanced stability settings"""
    
    # Model settings
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # model_name = 'gpt2'  # For testing
    
    hidden_size = 4096
    reasoning_dim = 192
    max_latent_steps = 4
    use_hierarchical_planning = True
    num_hierarchy_levels = 2
    recurrent_hidden_size = 384
    memory_size = 384
    dropout_rate = 0.1
    
    # Training settings - MORE CONSERVATIVE FOR STABILITY
    num_epochs = 3
    batch_size = 1
    gradient_accumulation_steps = 8  # Reduced for stability
    effective_batch_size = batch_size * gradient_accumulation_steps
    learning_rate = 1e-5      # Reduced
    navigator_lr = 2e-5       # Reduced
    planner_lr = 1.5e-5       # Reduced
    value_lr = 1e-5           # Reduced
    warmup_steps = 500        # Increased
    max_grad_norm = 0.1       # Much more conservative
    
    # PPO settings - MORE CONSERVATIVE
    ppo_epochs = 1            # Reduced
    ppo_batch_size = 16
    cliprange = 0.05          # Much smaller
    cliprange_value = 0.05
    value_coef = 0.5
    entropy_coef = 0.005      # Reduced
    target_kl = 0.01
    gamma = 0.99
    gae_lambda = 0.95
    
    # Anti-mode-collapse settings
    trajectory_diversity_weight = 0.5
    min_trajectory_bonus = 2.0
    optimal_trajectory_range = (2, 4)
    trajectory_std_target = 1.0
    curriculum_stage_steps = 500
    
    # Replay buffer
    buffer_capacity = 1024
    buffer_update_frequency = 64
    
    # Reward settings
    correct_answer_reward = 10.0
    partial_correct_reward = 5.0
    wrong_answer_penalty = -1.0
    trajectory_length_penalty = -0.05
    planning_alignment_bonus = 1.0
    
    # Data settings
    dataset_size = 1000
    eval_samples = 25
    
    # Logging and saving
    log_interval = 10
    save_interval = 500
    eval_interval = 200
    
    # Paths
    checkpoint_dir = 'checkpoints_coconut_optimized'
    log_file = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    # Device and precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = False
    use_bfloat16 = True
    load_in_8bit = False
    
    # Memory optimization
    freeze_layers = 30
    gradient_checkpointing = True
    max_seq_length = 384
    clear_cache_frequency = 10

# ============================================================
# SAFE PPO REPLAY BUFFER WITH NAN PROTECTION
# ============================================================

class SafePPOReplayBuffer(PPOReplayBuffer):
    """PPO Replay Buffer with NaN protection"""
    
    def add_step(self, state, action, log_prob, value, reward, done, info):
        """Add step with NaN checking"""
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        
        # Check and fix NaN values
        if torch.isnan(state).any():
            logger.debug("NaN in state, replacing with zeros")
            state = torch.zeros_like(state)
        if torch.isnan(log_prob).any():
            logger.debug("NaN in log_prob, replacing with log(0.5)")
            log_prob = torch.log(torch.tensor(0.5))
        if torch.isnan(value).any():
            logger.debug("NaN in value, replacing with 0")
            value = torch.tensor(0.0)
        if math.isnan(reward) if isinstance(reward, float) else torch.isnan(reward).any():
            logger.debug("NaN in reward, replacing with 0")
            reward = 0.0
        
        # Call parent method
        super().add_step(state, action, log_prob, value, reward, done, info)
    
    def get_all_batches(self, batch_size):
        """Get batches with NaN filtering"""
        batches = super().get_all_batches(batch_size)
        
        # Filter out batches with NaN
        clean_batches = []
        for batch in batches:
            has_nan = False
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    has_nan = True
                    break
            
            if not has_nan:
                clean_batches.append(batch)
            else:
                logger.debug("Filtered out batch with NaN values")
        
        return clean_batches

# ============================================================
# ENHANCED METRICS LOGGER
# ============================================================

class MetricsLogger:
    """Enhanced logging with trajectory diversity tracking"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics_history = {}
        self.trajectory_length_history = deque(maxlen=1000)
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'metric', 'value'])
    
    def log(self, epoch: int, step: int, metrics: Dict):
        """Log metrics to CSV"""
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for metric, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                writer.writerow([timestamp, epoch, step, metric, float(value)])
                
                if metric not in self.metrics_history:
                    self.metrics_history[metric] = []
                self.metrics_history[metric].append(float(value))
    
    def add_trajectory_length(self, length: int):
        """Track trajectory length for diversity metrics"""
        self.trajectory_length_history.append(length)
    
    def get_trajectory_diversity(self) -> Dict:
        """Calculate trajectory length diversity metrics"""
        if len(self.trajectory_length_history) < 10:
            return {'mean': 2.0, 'std': 0.5, 'unique': 2}
        
        lengths = list(self.trajectory_length_history)
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'unique': len(set(lengths)),
            'min': min(lengths),
            'max': max(lengths)
        }

# ============================================================
# GRADIENT HEALTH MONITORING
# ============================================================

def check_gradient_health(model, step):
    """Monitor gradient health and fix NaN gradients"""
    total_norm = 0.0
    param_count = 0
    nan_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            # Check for NaN
            if torch.isnan(param.grad).any():
                nan_count += 1
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)
            
            # Calculate norm
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** 0.5
    
    if nan_count > 0:
        logger.warning(f"Step {step}: Found NaN in {nan_count}/{param_count} gradients, replaced with zeros")
    
    if total_norm > 100:
        logger.warning(f"Step {step}: Large gradient norm: {total_norm:.2f}")
    
    return total_norm

# ============================================================
# STABLE MODEL INITIALIZATION
# ============================================================

def initialize_model_stable(model, config):
    """Enhanced initialization with numerical stability"""
    
    with torch.no_grad():
        # Navigator initialization
        for name, module in model.navigator.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        # Value network initialization
        for name, module in model.value_network.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Planner initialization
        if model.planner is not None:
            for name, module in model.planner.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.001)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
        
        # Special handling for continue head
        if hasattr(model.navigator, 'continue_head'):
            for module in model.navigator.continue_head.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.fill_(0.0)
                    if module.bias is not None and module.out_features == 2:
                        module.bias.data[0] = -0.2  # Stop
                        module.bias.data[1] = 0.2   # Continue
    
    return model

def patch_navigator_stable(model):
    """Patch navigator with numerical stability safeguards"""
    
    original_navigate = model.navigator.navigate
    
    def stable_navigate(hidden_state, deterministic=False, return_policy_info=False):
        """Numerically stable navigation with NaN prevention"""
        
        # Input validation and stabilization
        if torch.isnan(hidden_state).any() or torch.isinf(hidden_state).any():
            logger.debug("Invalid values in navigate input, fixing...")
            hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Normalize hidden state to prevent explosion
        hidden_norm = torch.norm(hidden_state, dim=-1, keepdim=True)
        if (hidden_norm > 10.0).any():
            hidden_state = hidden_state * 10.0 / (hidden_norm + 1e-8)
        
        # Call original method with try-catch
        try:
            output = original_navigate(hidden_state, deterministic, return_policy_info)
        except (ValueError, RuntimeError) as e:
            if "nan" in str(e).lower() or "invalid" in str(e).lower():
                logger.debug(f"Navigation failed with {e}, returning safe defaults")
                # Return safe default values
                output = {
                    'continue_prob': torch.tensor(0.5, device=hidden_state.device),
                    'action': torch.tensor(1, device=hidden_state.device),
                    'hidden_state': torch.zeros_like(hidden_state),
                    'value': torch.tensor(0.0, device=hidden_state.device),
                    'log_prob': torch.log(torch.tensor(0.5, device=hidden_state.device)),
                    'entropy': torch.tensor(0.693, device=hidden_state.device)
                }
                if return_policy_info:
                    output.update({
                        'policy_info': {'continue_probs': torch.tensor([0.5, 0.5], device=hidden_state.device)}
                    })
                return output
            else:
                raise e
        
        # Validate and fix output
        if isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        logger.debug(f"Invalid values in navigate output '{key}', fixing...")
                        if key == 'continue_prob':
                            output[key] = torch.tensor(0.5, device=value.device)
                        elif key == 'value':
                            output[key] = torch.zeros_like(value)
                        elif key == 'log_prob':
                            output[key] = torch.log(torch.tensor(0.5, device=value.device))
                        elif key == 'entropy':
                            output[key] = torch.tensor(0.693, device=value.device)
                        elif key == 'hidden_state':
                            output[key] = torch.zeros_like(value)
                        else:
                            output[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output
    
    model.navigator.navigate = stable_navigate
    return model

# ============================================================
# OPTIMIZED PPO TRAINER WITH STABILITY
# ============================================================

class OptimizedPPOTrainer:
    """PPO Trainer with anti-mode-collapse and numerical stability"""
    
    def __init__(self, model: CoconutPPO, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        
        # Curriculum learning stage
        self.curriculum_stage = 0
        
        # Create optimizers
        self._create_optimizers()
        
        # Create schedulers
        self._create_schedulers()
        
        # Safe PPO replay buffer
        self.replay_buffer = SafePPOReplayBuffer(
            capacity=config.buffer_capacity,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device='cpu'  # Store on CPU
        )
        
        # No GradScaler needed for bfloat16
        self.scaler = None
        logger.info("Using BFloat16 precision (no GradScaler needed)")
        
        # Metrics
        self.logger = MetricsLogger(config.log_file)
        self.global_step = 0
        self.best_eval_accuracy = 0
        
        # Gradient accumulation
        self.accumulation_steps = config.gradient_accumulation_steps
        self.accumulated_steps = 0
    
    def _create_optimizers(self):
        """Create optimizers with settings for stability"""
        # Navigator parameters
        navigator_params = list(self.model.navigator.parameters())
        
        # Planner parameters
        planner_params = []
        if self.model.planner is not None:
            planner_params = list(self.model.planner.parameters())
        
        # Value network parameters
        value_params = list(self.model.value_network.parameters())
        
        # Base model parameters (only unfrozen layers)
        base_params = []
        for name, param in self.model.base_model.named_parameters():
            if param.requires_grad:
                base_params.append(param)
        
        # Create optimizers with conservative settings
        eps = 1e-8
        
        self.optimizer_base = AdamW(
            base_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
            eps=eps,
            betas=(0.9, 0.999)
        )
        
        self.optimizer_navigator = AdamW(
            navigator_params,
            lr=self.config.navigator_lr,
            weight_decay=0.01,
            eps=eps,
            betas=(0.9, 0.999)
        )
        
        self.optimizer_value = AdamW(
            value_params,
            lr=self.config.value_lr,
            weight_decay=0.01,
            eps=eps,
            betas=(0.9, 0.999)
        )
        
        if planner_params:
            self.optimizer_planner = AdamW(
                planner_params,
                lr=self.config.planner_lr,
                weight_decay=0.01,
                eps=eps,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer_planner = None
        
        self.optimizers = [self.optimizer_base, self.optimizer_navigator, self.optimizer_value]
        if self.optimizer_planner:
            self.optimizers.append(self.optimizer_planner)
    
    def _create_schedulers(self):
        """Create learning rate schedulers with warmup"""
        total_steps = self.config.num_epochs * (self.config.dataset_size // self.config.effective_batch_size)
        
        self.schedulers = []
        for optimizer in self.optimizers:
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # Start very small
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
            
            # Cosine annealing after warmup
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=1e-7
            )
            
            # Sequential scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_steps]
            )
            
            self.schedulers.append(scheduler)
    
    def update_curriculum_stage(self):
        """Update curriculum learning stage"""
        if self.global_step < self.config.curriculum_stage_steps:
            self.curriculum_stage = 0
        elif self.global_step < 2 * self.config.curriculum_stage_steps:
            self.curriculum_stage = 1
        else:
            self.curriculum_stage = 2
    
    def compute_reward_with_diversity(
        self,
        generated_text: str,
        true_answer: str,
        trajectory: Dict,
        batch_trajectories: List[Dict] = None
    ) -> Tuple[float, bool]:
        """Enhanced reward computation with diversity bonus"""
        
        # Extract answer
        gen_answer = self.extract_answer(generated_text)
        correct = gen_answer == true_answer if gen_answer and true_answer else False
        
        # Base reward
        if correct:
            reward = self.config.correct_answer_reward
        elif gen_answer and true_answer and len(gen_answer) == len(true_answer):
            reward = self.config.partial_correct_reward
        else:
            reward = self.config.wrong_answer_penalty
        
        # Trajectory length reward with curriculum learning
        traj_len = len(trajectory.get('states', []))
        
        if traj_len > 0:
            self.logger.add_trajectory_length(traj_len)
            
            if self.curriculum_stage == 0:
                if traj_len >= 2:
                    reward += self.config.min_trajectory_bonus * (1 + traj_len * 0.2)
                else:
                    reward -= 2.0
            elif self.curriculum_stage == 1:
                if self.config.optimal_trajectory_range[0] <= traj_len <= self.config.optimal_trajectory_range[1]:
                    reward += 1.0
                elif traj_len == 1:
                    reward -= 0.5
                else:
                    reward += self.config.trajectory_length_penalty * abs(traj_len - 3)
            else:
                if self.config.optimal_trajectory_range[0] <= traj_len <= self.config.optimal_trajectory_range[1]:
                    reward += 0.5
                if traj_len == 1:
                    reward -= 0.2
                elif traj_len > 5:
                    reward -= 0.1 * (traj_len - 5)
        
        # Diversity bonus
        if batch_trajectories and len(batch_trajectories) > 1:
            lengths = [len(t.get('states', [])) for t in batch_trajectories]
            if len(set(lengths)) > 1:
                reward += self.config.trajectory_diversity_weight
        
        # Planning alignment bonus
        if trajectory.get('planning_info'):
            plan_info = trajectory['planning_info']
            if 'depth_reached' in plan_info:
                depth = plan_info['depth_reached']
                if abs(depth - traj_len) <= 1:
                    reward += self.config.planning_alignment_bonus
        
        return reward, correct
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from text"""
        patterns = [
            r'####\s*(\d+)',
            r'answer is\s*(\d+)',
            r'answer:\s*(\d+)',
            r'equals?\s*(\d+)',
            r'=\s*(\d+)\s*(?:\.|$)'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        return None
    
    def train_step(self, batch_data: List[Dict]) -> Dict:
        """Training step with stability measures"""
        self.model.train()
        
        step_metrics = {
            'loss': [],
            'accuracy': [],
            'rewards': [],
            'trajectory_lengths': []
        }
        
        batch_trajectories = []
        
        for item_idx, item in enumerate(batch_data):
            if item_idx > 0 and item_idx % 2 == 0 and self.config.device == 'cuda':
                torch.cuda.empty_cache()
            
            question = item['question']
            answer_text = item['answer']
            true_answer = self.extract_answer(answer_text)
            
            prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=min(256, self.config.max_seq_length // 2),
                padding=True
            ).to(self.device)
            
            answer_tokens = self.tokenizer(
                answer_text,
                return_tensors='pt',
                truncation=True,
                max_length=min(128, self.config.max_seq_length // 2),
                padding=False
            )['input_ids'].to(self.device)
            
            full_tokens = torch.cat([inputs['input_ids'], answer_tokens], dim=1)
            
            if full_tokens.shape[1] > self.config.max_seq_length:
                full_tokens = full_tokens[:, :self.config.max_seq_length]
            
            labels = full_tokens.clone()
            labels[:, :inputs['input_ids'].shape[1]] = -100
            
            try:
                full_attention_mask = torch.ones_like(full_tokens, dtype=torch.long)
                
                with autocast(device_type='cuda' if self.config.device == 'cuda' else 'cpu',
                            dtype=torch.bfloat16 if self.config.use_bfloat16 else None,
                            enabled=self.config.use_bfloat16):
                    outputs = self.model(
                        input_ids=full_tokens,
                        attention_mask=full_attention_mask,
                        labels=labels,
                        return_trajectory=True,
                        use_planning=self.config.use_hierarchical_planning,
                        deterministic=False
                    )
                
                loss = outputs.get('loss')
                trajectory = outputs.get('trajectory', {})
                
                if isinstance(trajectory, list):
                    trajectory = trajectory[0]
                batch_trajectories.append(trajectory)
                
                del outputs
                
                if loss is not None:
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Detected inf/nan loss, skipping item {item_idx}")
                        del loss
                        continue
                    
                    # Scale loss conservatively
                    loss = loss / (self.config.gradient_accumulation_steps * 2.0)
                    loss.backward()
                    
                    # Check gradient health
                    grad_norm = check_gradient_health(self.model, self.global_step)
                    
                    step_metrics['loss'].append(loss.item() * self.config.gradient_accumulation_steps * 2.0)
                    del loss
                
                with torch.no_grad():
                    generation = self.model.base_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=min(100, self.config.max_seq_length // 4),
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(generation[0], skip_special_tokens=True)
                    del generation
                
                reward, correct = self.compute_reward_with_diversity(
                    generated_text, true_answer, trajectory, batch_trajectories
                )
                
                # Add to replay buffer with NaN checking
                if 'states' in trajectory and trajectory['states']:
                    max_steps = min(3, len(trajectory['states']))
                    for i in range(max_steps):
                        state = trajectory['states'][i]
                        action = trajectory['actions'][i] if i < len(trajectory['actions']) else torch.tensor(0)
                        log_prob = trajectory['log_probs'][i] if i < len(trajectory['log_probs']) else torch.tensor(0)
                        value = trajectory['values'][i] if i < len(trajectory['values']) else torch.tensor(0)
                        
                        step_reward = reward / max_steps
                        done = (i == max_steps - 1)
                        
                        self.replay_buffer.add_step(
                            state=state.detach().cpu(),
                            action=action.detach().cpu() if hasattr(action, 'detach') else action,
                            log_prob=log_prob.detach().cpu() if hasattr(log_prob, 'detach') else log_prob,
                            value=value.detach().cpu() if hasattr(value, 'detach') else value,
                            reward=step_reward,
                            done=done,
                            info={}
                        )
                
                step_metrics['accuracy'].append(float(correct))
                step_metrics['rewards'].append(reward)
                step_metrics['trajectory_lengths'].append(len(trajectory.get('states', [])))
                
                del trajectory
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM in training step {item_idx}: {e}")
                    if self.config.device == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                logger.warning(f"Error in training step {item_idx}: {e}")
                continue
            finally:
                for var in ['full_tokens', 'labels', 'full_attention_mask', 'inputs', 'answer_tokens']:
                    if var in locals():
                        del locals()[var]
        
        # Update after accumulation
        self.accumulated_steps += 1
        if self.accumulated_steps >= self.config.gradient_accumulation_steps:
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            for optimizer in self.optimizers:
                optimizer.step()
            
            # Zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            
            # Step schedulers
            for scheduler in self.schedulers:
                scheduler.step()
            
            self.accumulated_steps = 0
            
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
        
        return step_metrics
    
    def ppo_update(self):
        """PPO update with comprehensive NaN protection"""
        if len(self.replay_buffer) < self.config.ppo_batch_size:
            return
        
        batches = self.replay_buffer.get_all_batches(self.config.ppo_batch_size)
        if not batches:
            return
        
        # Limit batches for memory
        batches = batches[:min(len(batches), 4)]
        
        for ppo_epoch in range(min(self.config.ppo_epochs, 2)):
            for batch in batches:
                try:
                    states = batch['states'].to(self.device)
                    actions = batch['actions'].to(self.device)
                    old_log_probs = batch['old_log_probs'].to(self.device)
                    advantages = batch['advantages'].to(self.device)
                    returns = batch['returns'].to(self.device)
                    values = batch['values'].to(self.device)
                    
                    # Process in chunks
                    chunk_size = min(8, len(states))
                    for i in range(0, len(states), chunk_size):
                        chunk_states = states[i:i+chunk_size]
                        chunk_actions = actions[i:i+chunk_size]
                        chunk_old_log_probs = old_log_probs[i:i+chunk_size]
                        chunk_advantages = advantages[i:i+chunk_size]
                        chunk_returns = returns[i:i+chunk_size]
                        chunk_values = values[i:i+chunk_size]
                        
                        # Forward with bfloat16
                        new_outputs = []
                        with autocast(device_type='cuda' if self.config.device == 'cuda' else 'cpu',
                                    dtype=torch.bfloat16 if self.config.use_bfloat16 else None,
                                    enabled=self.config.use_bfloat16):
                            for state in chunk_states:
                                nav_output = self.model.navigator.navigate(
                                    state,
                                    deterministic=False,
                                    return_policy_info=True
                                )
                                
                                # Validate output
                                if (nav_output is not None and 
                                    not torch.isnan(nav_output.get('log_prob', torch.tensor(float('nan')))).any()):
                                    new_outputs.append(nav_output)
                        
                        # Skip if no valid outputs
                        if not new_outputs:
                            logger.debug("No valid outputs from navigator, skipping chunk")
                            continue
                        
                        new_log_probs = torch.stack([out['log_prob'] for out in new_outputs])
                        new_values = torch.stack([out['value'] for out in new_outputs])
                        entropies = torch.stack([out['entropy'] for out in new_outputs])
                        
                        # Adjust chunk sizes to match valid outputs
                        valid_count = len(new_outputs)
                        chunk_old_log_probs = chunk_old_log_probs[:valid_count]
                        chunk_advantages = chunk_advantages[:valid_count]
                        chunk_returns = chunk_returns[:valid_count]
                        chunk_values = chunk_values[:valid_count]
                        
                        # PPO loss with stability
                        ratio = torch.exp(new_log_probs - chunk_old_log_probs)
                        ratio = torch.clamp(ratio, 0.01, 100.0)
                        
                        surr1 = ratio * chunk_advantages
                        surr2 = torch.clamp(ratio, 1 - self.config.cliprange, 1 + self.config.cliprange) * chunk_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        value_pred_clipped = chunk_values + torch.clamp(
                            new_values - chunk_values,
                            -self.config.cliprange_value,
                            self.config.cliprange_value
                        )
                        value_losses = (new_values - chunk_returns) ** 2
                        value_losses_clipped = (value_pred_clipped - chunk_returns) ** 2
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                        
                        entropy_loss = -entropies.mean()
                        
                        loss = (
                            policy_loss +
                            self.config.value_coef * value_loss +
                            self.config.entropy_coef * entropy_loss
                        )
                        
                        # Final NaN check
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.debug("NaN/Inf loss detected, skipping update")
                            continue
                        
                        loss = loss / max(1, len(range(0, len(states), chunk_size)))
                        loss.backward()
                        
                        # Clean up
                        del new_outputs, new_log_probs, new_values, entropies
                        del ratio, surr1, surr2, policy_loss, value_loss, entropy_loss, loss
                    
                    # Check gradient health before update
                    grad_norm = check_gradient_health(self.model, self.global_step)
                    
                    # Update weights
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    for optimizer in self.optimizers:
                        optimizer.step()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad(set_to_none=True)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM during PPO: {e}")
                        if self.config.device == 'cuda':
                            torch.cuda.empty_cache()
                        for optimizer in self.optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        break
                    else:
                        raise e
        
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
    
    def train(self, dataset):
        """Main training loop with stability"""
        logger.info("🚀 Starting Optimized COCONUT Training with Stability Measures")
        logger.info(f"Configuration:")
        logger.info(f"  • Batch size: {self.config.batch_size}")
        logger.info(f"  • Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  • Max latent steps: {self.config.max_latent_steps}")
        logger.info(f"  • Using BFloat16: {self.config.use_bfloat16}")
        logger.info(f"  • Conservative learning rates")
        logger.info(f"  • Enhanced stability measures")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"📅 Epoch {epoch+1}/{self.config.num_epochs}")
            
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            indices = np.random.permutation(len(dataset))[:self.config.dataset_size]
            
            progress_bar = tqdm(
                range(0, len(indices), self.config.batch_size),
                desc=f"Training Epoch {epoch+1}"
            )
            
            epoch_metrics = {
                'loss': [],
                'accuracy': [],
                'rewards': [],
                'trajectory_lengths': []
            }
            
            for batch_idx, start_idx in enumerate(progress_bar):
                self.update_curriculum_stage()
                
                end_idx = min(start_idx + self.config.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_data = [dataset[int(idx)] for idx in batch_indices]
                
                step_metrics = self.train_step(batch_data)
                
                for key in step_metrics:
                    if step_metrics[key]:
                        epoch_metrics[key].extend(step_metrics[key])
                
                # PPO update
                if len(self.replay_buffer) >= self.config.buffer_update_frequency:
                    if self.config.device == 'cuda':
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        if free_memory > 2e9:
                            self.ppo_update()
                    else:
                        self.ppo_update()
                    
                    if len(self.replay_buffer) >= self.config.buffer_capacity * 0.8:
                        self.replay_buffer.buffer = list(self.replay_buffer.buffer)[len(self.replay_buffer)//2:]
                
                # Update progress bar
                if epoch_metrics['accuracy']:
                    diversity_stats = self.logger.get_trajectory_diversity()
                    recent_acc = np.mean(epoch_metrics['accuracy'][-100:])
                    recent_reward = np.mean(epoch_metrics['rewards'][-100:])
                    recent_traj_len = np.mean(epoch_metrics['trajectory_lengths'][-100:]) if epoch_metrics['trajectory_lengths'] else 0
                    
                    progress_bar.set_postfix({
                        'stage': self.curriculum_stage,
                        'acc': f"{recent_acc:.2%}",
                        'reward': f"{recent_reward:.2f}",
                        'traj_len': f"{recent_traj_len:.1f}",
                        'traj_std': f"{diversity_stats['std']:.2f}",
                        'buffer': len(self.replay_buffer)
                    })
                
                # Log metrics
                if self.global_step > 0 and self.global_step % self.config.log_interval == 0 and epoch_metrics['loss']:
                    diversity_stats = self.logger.get_trajectory_diversity()
                    log_data = {
                        'loss': np.mean(epoch_metrics['loss'][-100:]),
                        'accuracy': np.mean(epoch_metrics['accuracy'][-100:]),
                        'reward': np.mean(epoch_metrics['rewards'][-100:]),
                        'trajectory_length': np.mean(epoch_metrics['trajectory_lengths'][-100:]) if epoch_metrics['trajectory_lengths'] else 0,
                        'trajectory_std': diversity_stats['std'],
                        'trajectory_unique': diversity_stats['unique'],
                        'curriculum_stage': self.curriculum_stage,
                        'buffer_size': len(self.replay_buffer)
                    }
                    self.logger.log(epoch, self.global_step, log_data)
                
                # Evaluate
                if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
                    eval_results = self.evaluate(dataset)
                    logger.info(f"📊 Evaluation: {eval_results}")
                    self.logger.log(epoch, self.global_step, eval_results)
                    
                    if eval_results['eval_accuracy'] > self.best_eval_accuracy:
                        self.best_eval_accuracy = eval_results['eval_accuracy']
                        self.save_checkpoint(epoch, is_best=True)
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(epoch)
                
                self.global_step += 1
                
                # Clear cache periodically
                if batch_idx > 0 and batch_idx % self.config.clear_cache_frequency == 0:
                    if self.config.device == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
            
            # Final PPO update
            if len(self.replay_buffer) > 0:
                self.ppo_update()
                self.replay_buffer.clear()
            
            # Epoch summary
            diversity_stats = self.logger.get_trajectory_diversity()
            logger.info(f"\n📊 Epoch {epoch+1} Summary:")
            logger.info(f"  • Accuracy: {np.mean(epoch_metrics['accuracy']) if epoch_metrics['accuracy'] else 0:.2%}")
            logger.info(f"  • Avg Reward: {np.mean(epoch_metrics['rewards']) if epoch_metrics['rewards'] else 0:.2f}")
            logger.info(f"  • Avg Loss: {np.mean(epoch_metrics['loss']) if epoch_metrics['loss'] else 0:.4f}")
            logger.info(f"  • Avg Trajectory Length: {np.mean(epoch_metrics['trajectory_lengths']) if epoch_metrics['trajectory_lengths'] else 0:.1f}")
            logger.info(f"  • Trajectory Diversity (std): {diversity_stats['std']:.2f}")
            logger.info(f"  • Unique Trajectory Lengths: {diversity_stats['unique']}")
            
            self.save_checkpoint(epoch)
            
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        logger.info("\n" + "="*60)
        logger.info("✅ Training Complete!")
        logger.info(f"🏆 Best Accuracy: {self.best_eval_accuracy:.2%}")
    
    def evaluate(self, dataset, num_samples: int = None) -> Dict:
        """Evaluate with diversity metrics"""
        self.model.eval()
        num_samples = num_samples or self.config.eval_samples
        
        eval_metrics = {
            'accuracy': [],
            'trajectory_length': [],
            'planning_depth': []
        }
        
        eval_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        with torch.no_grad():
            for idx in tqdm(eval_indices, desc="Evaluating"):
                if self.config.device == 'cuda':
                    torch.cuda.empty_cache()
                
                item = dataset[int(idx)]
                question = item['question']
                answer_text = item['answer']
                true_answer = self.extract_answer(answer_text)
                
                prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=min(256, self.config.max_seq_length // 2)
                ).to(self.device)
                
                try:
                    with autocast(device_type='cuda' if self.config.device == 'cuda' else 'cpu',
                                dtype=torch.bfloat16 if self.config.use_bfloat16 else None,
                                enabled=self.config.use_bfloat16):
                        outputs = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            return_trajectory=True,
                            use_planning=self.config.use_hierarchical_planning,
                            deterministic=True
                        )
                    
                    trajectory = outputs.get('trajectory', {})
                    if isinstance(trajectory, list):
                        trajectory = trajectory[0]
                    
                    del outputs
                    
                    generation = self.model.base_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=min(100, self.config.max_seq_length // 4),
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(generation[0], skip_special_tokens=True)
                    gen_answer = self.extract_answer(generated_text)
                    
                    del generation
                    
                    correct = gen_answer == true_answer if gen_answer and true_answer else False
                    eval_metrics['accuracy'].append(float(correct))
                    
                    traj_len = len(trajectory.get('states', []))
                    eval_metrics['trajectory_length'].append(traj_len)
                    
                    if trajectory.get('planning_info'):
                        plan_depth = trajectory['planning_info'].get('depth_reached', 0)
                        eval_metrics['planning_depth'].append(plan_depth)
                    
                    del trajectory
                    
                except Exception as e:
                    logger.warning(f"Evaluation error: {e}")
                    continue
                finally:
                    if 'inputs' in locals():
                        del inputs
        
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Calculate metrics
        if eval_metrics['trajectory_length']:
            traj_std = np.std(eval_metrics['trajectory_length'])
            unique_lengths = len(set(eval_metrics['trajectory_length']))
        else:
            traj_std = 0
            unique_lengths = 0
        
        summary = {
            'eval_accuracy': np.mean(eval_metrics['accuracy']) if eval_metrics['accuracy'] else 0,
            'eval_avg_trajectory': np.mean(eval_metrics['trajectory_length']) if eval_metrics['trajectory_length'] else 0,
            'eval_trajectory_std': traj_std,
            'eval_unique_lengths': unique_lengths,
            'eval_avg_planning_depth': np.mean(eval_metrics['planning_depth']) if eval_metrics['planning_depth'] else 0
        }
        
        self.model.train()
        return summary
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
            'scheduler_states': [sched.state_dict() for sched in self.schedulers],
            'best_eval_accuracy': self.best_eval_accuracy,
            'curriculum_stage': self.curriculum_stage,
            'config': vars(self.config)
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            logger.info(f"💾 Saving best model with accuracy: {self.best_eval_accuracy:.2%}")
        else:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{self.global_step}.pt')
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main training function"""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1e9
        
        print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  • Total VRAM: {total_memory:.1f} GB")
        print(f"  • Available VRAM: {(gpu_props.total_memory - torch.cuda.memory_allocated(0)) / 1e9:.1f} GB")
    
    print("="*60)
    print("🥥 Optimized COCONUT Training with Stability Measures")
    print("="*60)
    
    # Configuration
    config = TrainingConfig()
    
    print("\n📊 Configuration Summary:")
    print(f"  • Model: {config.model_name}")
    print(f"  • Max latent steps: {config.max_latent_steps}")
    print(f"  • Reasoning dim: {config.reasoning_dim}")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  • Using BFloat16: {config.use_bfloat16}")
    print(f"  • Learning rate: {config.learning_rate}")
    print(f"  • Max grad norm: {config.max_grad_norm}")
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load base model
    print("🤖 Loading base model...")
    
    # Adjust for model type
    if 'gpt2' in config.model_name.lower():
        config.hidden_size = 768
        config.freeze_layers = min(config.freeze_layers, 10)
        print("  • Detected GPT-2 model, adjusted hidden_size to 768")
    
    # Load with bfloat16
    load_kwargs = {
        'use_cache': False,
        'torch_dtype': torch.bfloat16 if config.use_bfloat16 and config.device == 'cuda' else torch.float32,
        'device_map': 'auto' if config.device == 'cuda' else None
    }
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs
        )
    
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Freeze layers
    if config.freeze_layers > 0:
        print(f"❄️ Freezing first {config.freeze_layers} layers...")
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
        else:
            print("  ⚠️ Could not identify model architecture")
            layers = []
        
        for i, layer in enumerate(layers):
            if i < config.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    # Create COCONUT model
    print("🥥 Creating COCONUT model...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with suppress_output():
            model = CoconutPPO(
                base_model=base_model,
                latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
                start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
                end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
                eos_token_id=tokenizer.eos_token_id,
                hidden_size=config.hidden_size,
                reasoning_dim=config.reasoning_dim,
                max_latent_steps=config.max_latent_steps,
                use_hierarchical_planning=config.use_hierarchical_planning,
                num_hierarchy_levels=config.num_hierarchy_levels,
                recurrent_hidden_size=config.recurrent_hidden_size,
                memory_size=config.memory_size,
                dropout_rate=config.dropout_rate
            )
    
    # Apply stable initialization
    print("  • Applying enhanced numerical stability initialization...")
    model = initialize_model_stable(model, config)
    
    # Patch navigator for stability
    print("  • Patching navigator with stability safeguards...")
    model = patch_navigator_stable(model)
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing and hasattr(model.base_model, 'gradient_checkpointing_enable'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with suppress_output():
                model.base_model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  • Total parameters: {total_params/1e9:.2f}B" if total_params > 1e9 else f"  • Total parameters: {total_params/1e6:.2f}M")
    print(f"  • Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"  • Percentage trainable: {trainable_params/total_params*100:.2f}%")
    
    # Load dataset
    print("\n📊 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    print(f"  • Dataset size: {len(dataset)}")
    print(f"  • Training samples per epoch: {config.dataset_size}")
    
    # Create trainer
    print("\n🎯 Initializing Stable PPO trainer...")
    trainer = OptimizedPPOTrainer(model, tokenizer, config)
    
    # Start training
    print("\n" + "="*60)
    print("🚀 Starting training with stability measures...")
    print("  • Conservative learning rates")
    print("  • Enhanced initialization")
    print("  • NaN prevention at all levels")
    print("  • Gradient health monitoring")
    print("="*60)
    
    trainer.train(dataset)
    
    print("\n🎉 Training complete!")
    print(f"📝 Logs saved to: {config.log_file}")
    print(f"💾 Checkpoints saved to: {config.checkpoint_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
