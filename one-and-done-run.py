"""
COCONUT Training v5 - FULL MODEL Training (All Layers Unfrozen)
Key features:
1. Training entire unfrozen network (all 32 layers)
2. Reduce navigator bias (was too strong)
3. Add stopping rewards
4. Variable trajectory targets
5. Better reward shaping for different lengths
6. Stochastic trajectory sampling
7. Adjusted learning rates for full model stability
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
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
# Fixed import statement
if 'get_ipython' in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ============================================================
# NAVIGATOR WITH BETTER STOPPING
# ============================================================

class ContinuousReasoningNavigator(nn.Module):
    """Navigator with improved stopping mechanism"""
    def __init__(
        self,
        hidden_size: int,
        reasoning_dim: int = 256,
        memory_size: int = 500,
        top_k_memory: int = 3,
        fusion_weight: float = 0.5,
        dropout_rate: float = 0.1,
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
        
        # Projection layers
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, reasoning_dim)
        )
        self.thought_projection = nn.Sequential(
            nn.Linear(reasoning_dim, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        
        # Policy heads - IMPROVED STOPPING HEAD
        self.continue_head = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(reasoning_dim // 2, 2)  # More complex decision
        )
        self.direction_head = nn.Linear(reasoning_dim, reasoning_dim)
        self.step_size_head = nn.Linear(reasoning_dim, 1)
        self.value_head = nn.Linear(reasoning_dim, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Memory bank
        self.register_buffer('memory_bank', torch.zeros(memory_size, reasoning_dim))
        self.register_buffer('memory_values', torch.full((memory_size,), -float('inf')))
        self.memory_ptr = 0
        self.memory_filled = False
        
        # Step counter for position encoding
        self.register_buffer('step_encoding', torch.zeros(1, reasoning_dim))
        
        if device:
            self.to(device)
        if dtype:
            self.to(dtype)
    
    def _retrieve_from_memory(self, reasoning_state: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve from memory bank"""
        if not self.memory_filled or self.top_k_memory <= 0:
            return None
            
        if reasoning_state.dim() == 1:
            reasoning_state = reasoning_state.unsqueeze(0)
        
        norm_state = F.normalize(reasoning_state, p=2, dim=-1)
        norm_bank = F.normalize(self.memory_bank, p=2, dim=-1)
        
        similarities = torch.matmul(norm_state, norm_bank.t()).squeeze(0)
        weighted_sims = similarities * (self.memory_values + 1e-8)
        
        valid_indices = weighted_sims > -float('inf')
        if valid_indices.sum() == 0:
            return None
            
        top_k = min(self.top_k_memory, valid_indices.sum().item())
        top_k_indices = torch.topk(weighted_sims[valid_indices], k=top_k, sorted=False).indices
        
        retrieved_states = self.memory_bank[top_k_indices]
        avg_retrieved = retrieved_states.mean(dim=0)
        
        return avg_retrieved
    
    def _write_to_memory(self, position: torch.Tensor, value: torch.Tensor):
        """Write to memory bank"""
        if position.dim() > 1:
            position = position.mean(dim=0)
        
        with torch.no_grad():
            self.memory_bank[self.memory_ptr] = position.detach().clone()
            self.memory_values[self.memory_ptr] = value.detach().mean().clone() if value.dim() > 0 else value.detach().clone()
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank.size(0)
        
        if self.memory_ptr >= self.top_k_memory:
            self.memory_filled = True
    
    def navigate(self, state: torch.Tensor, step_num: int = 0, 
                return_policy_info: bool = True,
                temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Navigate with step-aware stopping"""
        state = state.to(self.continue_head[0].weight.device)
        state = state.to(self.continue_head[0].weight.dtype)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # Project to reasoning space
        reasoning_state = self.state_projection(state)
        
        # Add step encoding to help with stopping decisions
        step_embed = torch.sin(torch.tensor(step_num * 0.5, device=state.device))
        reasoning_state = reasoning_state + 0.1 * step_embed
        
        if self.training:
            reasoning_state = self.dropout(reasoning_state)
        
        # Memory retrieval
        retrieved = self._retrieve_from_memory(reasoning_state)
        if retrieved is not None:
            fused_state = (1 - self.fusion_weight) * reasoning_state + self.fusion_weight * retrieved.unsqueeze(0)
            reasoning_state = fused_state
        
        # Get policy outputs with temperature
        continue_logits = self.continue_head(reasoning_state) / temperature
        continue_probs = F.softmax(continue_logits, dim=-1)
        
        # Sample or argmax based on training/eval
        if self.training:
            continue_dist = torch.distributions.Categorical(continue_probs)
            continue_action = continue_dist.sample()
        else:
            continue_action = continue_probs.argmax(dim=-1)
        
        # Generate direction and step
        direction_raw = self.direction_head(reasoning_state)
        direction = F.normalize(direction_raw, p=2, dim=-1)
        step_size = torch.sigmoid(self.step_size_head(reasoning_state)) * 2.0
        value = self.value_head(reasoning_state)
        
        # Move in reasoning space
        next_position = reasoning_state + step_size * direction
        latent_thought = self.thought_projection(next_position)
        
        # Write to memory
        self._write_to_memory(next_position, value)
        
        # Handle stop condition
        if single_input:
            stop_condition = continue_action.item() == 1
        else:
            stop_condition = (continue_action == 1)
        
        result = {
            'latent_thought': latent_thought.squeeze(0) if single_input else latent_thought,
            'stop': stop_condition,
            'position': next_position.squeeze(0) if single_input else next_position,
            'continue_prob': continue_probs[:, 0].item() if single_input else continue_probs[:, 0]
        }
        
        if return_policy_info:
            if self.training:
                log_prob = continue_dist.log_prob(continue_action)
                entropy = continue_dist.entropy()
            else:
                log_prob = torch.log(continue_probs.gather(1, continue_action.unsqueeze(1)).squeeze())
                entropy = -(continue_probs * torch.log(continue_probs + 1e-8)).sum(dim=-1)
            
            result.update({
                'action': continue_action.squeeze(0) if single_input else continue_action,
                'log_prob': log_prob.squeeze(0) if single_input else log_prob,
                'value': value.squeeze() if single_input else value.squeeze(-1),
                'entropy': entropy.squeeze(0) if single_input else entropy
            })
        
        return result

class CoconutPPO(nn.Module):
    """COCONUT with variable trajectory lengths"""
    
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
        
        self.navigator = ContinuousReasoningNavigator(
            hidden_size=hidden_size,
            reasoning_dim=reasoning_dim,
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
        return_trajectory: bool = True,
        target_trajectory: int = None,  # NEW: target length
        temperature: float = 1.0  # NEW: temperature for sampling
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with variable trajectory"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Initialize trajectory storage
        batch_trajectories = []
        batch_latent_sequences = []
        
        for b in range(batch_size):
            trajectory = {
                'states': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'latent_embeds': [],
                'entropy': [],
                'continue_probs': []
            }
            
            # Get initial hidden state
            with torch.no_grad():
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[b:b+1],
                    attention_mask=attention_mask[b:b+1] if attention_mask is not None else None,
                    output_hidden_states=True,
                    use_cache=False
                )
                current_state = outputs.hidden_states[-1].mean(dim=1).squeeze(0).detach()
            
            # Navigate through reasoning space
            latent_sequence = []
            actual_stopped = False
            
            for step in range(self.max_latent_steps):
                # Navigate with step awareness
                nav_output = self.navigator.navigate(
                    current_state, 
                    step_num=step,
                    return_policy_info=True,
                    temperature=temperature
                )
                
                # Store trajectory info
                if return_trajectory:
                    trajectory['states'].append(current_state.detach())
                    trajectory['actions'].append(nav_output.get('action'))
                    trajectory['log_probs'].append(nav_output.get('log_prob'))
                    trajectory['values'].append(nav_output.get('value'))
                    trajectory['entropy'].append(nav_output.get('entropy', torch.tensor(0.0)))
                    trajectory['continue_probs'].append(nav_output.get('continue_prob', 0.5))
                
                # Check stopping - but respect target if provided
                should_stop = nav_output['stop'] if isinstance(nav_output['stop'], bool) else nav_output['stop'].item()
                
                # If we have a target trajectory, override stopping
                if target_trajectory is not None:
                    if step < target_trajectory - 1:
                        should_stop = False  # Force continue
                    elif step >= target_trajectory - 1:
                        should_stop = True  # Force stop at target
                
                if should_stop:
                    actual_stopped = True
                    break
                
                # Generate latent thought
                latent_thought = nav_output['latent_thought']
                latent_sequence.append(latent_thought)
                
                if return_trajectory:
                    trajectory['latent_embeds'].append(latent_thought.detach())
                
                current_state = latent_thought
            
            # Store whether it stopped naturally
            trajectory['natural_stop'] = actual_stopped
            
            batch_trajectories.append(trajectory)
            batch_latent_sequences.append(latent_sequence)
        
        # Process through model with latent thoughts
        max_latent_len = max(len(seq) for seq in batch_latent_sequences) if batch_latent_sequences else 0
        
        if max_latent_len > 0:
            # Adjust labels for latent thoughts
            if labels is not None:
                batch_size, seq_len = labels.shape
                new_labels = torch.full(
                    (batch_size, seq_len + max_latent_len),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                new_labels[:, 0] = labels[:, 0]
                new_labels[:, max_latent_len + 1:] = labels[:, 1:]
                labels = new_labels
            
            # Pad latent sequences
            padded_latent_sequences = []
            for seq in batch_latent_sequences:
                if len(seq) == 0:
                    dummy_thought = torch.zeros(self.hidden_size, device=device, dtype=inputs_embeds.dtype)
                    padding = [dummy_thought for _ in range(max_latent_len)]
                    padded_seq = padding
                elif len(seq) < max_latent_len:
                    padding = [torch.zeros_like(seq[0]) for _ in range(max_latent_len - len(seq))]
                    padded_seq = seq + padding
                else:
                    padded_seq = seq
                
                stacked_seq = torch.stack(padded_seq, dim=0)
                padded_latent_sequences.append(stacked_seq)
            
            latent_embeds = torch.stack(padded_latent_sequences, dim=0)
            
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
        
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'trajectory': batch_trajectories[0] if batch_trajectories else {
                'states': [], 'actions': [], 'log_probs': [], 
                'values': [], 'latent_embeds': [], 'entropy': []
            }
        }

# ============================================================
# ANSWER PARSING AND GENERATION
# ============================================================

def parse_final_answer(text):
    """Extract numerical answer from text"""
    if not text:
        return None
    
    # GSM8K format
    gsm_match = re.search(r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if gsm_match:
        answer = gsm_match.group(1).strip().replace(',', '')
        try:
            return float(answer) if '.' in answer else int(answer)
        except:
            pass
    
    # Look for last number
    numbers = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        answer = numbers[-1].replace(',', '')
        try:
            return float(answer) if '.' in answer else int(answer)
        except:
            pass
    
    return None

def generate_simple(model, tokenizer, prompt, max_new_tokens=100):
    """Simple generation without trajectories"""
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(model.base_model.device)
        
        generated = model.base_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        input_len = inputs['input_ids'].shape[1]
        response = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
    
    model.train()
    return response

def check_answer_correctness(pred_text, true_text, tolerance=1e-5):
    """Check if answers match"""
    pred_answer = parse_final_answer(pred_text)
    true_answer = parse_final_answer(true_text)
    
    if pred_answer is None or true_answer is None:
        return False
    
    try:
        pred_num = float(pred_answer)
        true_num = float(true_answer)
        
        if abs(pred_num - true_num) < tolerance:
            return True
        
        if abs(true_num) > 1:
            rel_diff = abs(pred_num - true_num) / abs(true_num)
            return rel_diff < 0.01
        
        return False
    except:
        return False

# ============================================================
# IMPROVED REWARD WITH STOPPING BONUS
# ============================================================

def compute_reward_v2(trajectory_length, correct, natural_stop, target_length=4):
    """Reward function that encourages proper stopping"""
    reward = 0.0
    
    # Correctness reward
    if correct:
        reward += 1.0
    else:
        reward -= 0.3
    
    # Length-based reward with peak at target
    if trajectory_length == 0:
        reward -= 1.5  # Heavy penalty for no reasoning
    elif trajectory_length == target_length:
        reward += 0.5  # Bonus for hitting target
    elif abs(trajectory_length - target_length) == 1:
        reward += 0.2  # Small bonus for being close
    elif trajectory_length > 6:
        reward -= 0.3 * (trajectory_length - 6)  # Penalty for too long
    
    # Bonus for natural stopping (not hitting max)
    if natural_stop and 2 <= trajectory_length <= 5:
        reward += 0.3  # Reward for deciding to stop appropriately
    
    return reward

# ============================================================
# MAIN TRAINING WITH VARIABLE TRAJECTORIES
# ============================================================

def train():
    # Configuration
    num_epochs = 5
    batch_size = 1
    gradient_accumulation_steps = 8
    max_length = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Learning rates (adjusted for full model training)
    learning_rate = 5e-6  # Reduced from 1e-5 for stability with full model
    navigator_lr = 1e-5   # Reduced from 1.5e-5
    
    # Trajectory settings
    MIN_TRAJECTORY = 2
    MAX_TRAJECTORY = 6
    TARGET_TRAJECTORY = 4  # Ideal length
    
    # Temperature for exploration
    INITIAL_TEMP = 1.5  # Start with more exploration
    FINAL_TEMP = 0.5    # End with more exploitation
    
    print("\n" + "="*60)
    print("COCONUT Training v5 - FULL MODEL (Unfrozen)")
    print("="*60)
    
    # Load model and tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    print("🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    
    print("🥥 Creating COCONUT model...")
    model = CoconutPPO(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=256,
        max_latent_steps=MAX_TRAJECTORY,
        dropout_rate=0.1
    )
    
    # MODERATE navigator bias (not too strong!)
    print("🎯 Setting MODERATE navigator bias...")
    with torch.no_grad():
        if hasattr(model.navigator, 'continue_head'):
            # Use sequential structure
            first_layer = model.navigator.continue_head[0]
            last_layer = model.navigator.continue_head[-1]
            if hasattr(last_layer, 'bias'):
                last_layer.bias.data[0] += 0.5  # Moderate continue bias
                last_layer.bias.data[1] -= 0.5  # Moderate stop bias
    
    # Training FULL unfrozen network
    print("🔥 Training FULL unfrozen network (all layers)...")
    # No freezing - all layers will be trained
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable: {trainable_params/1e6:.2f}M / {total_params/1e9:.2f}B (FULL MODEL)")
    
    model.to(device)
    
    # Optimizer
    navigator_params = list(model.navigator.parameters())
    navigator_param_ids = {id(p) for p in navigator_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in navigator_param_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': navigator_params, 'lr': navigator_lr, 'weight_decay': 0.001},
        {'params': base_params, 'lr': learning_rate, 'weight_decay': 0.01}
    ], betas=(0.9, 0.95))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Load dataset
    print("📊 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    val_dataset = load_dataset("gsm8k", "main", split="test")
    
    print("\n🚀 Starting FULL MODEL training with VARIABLE trajectories!")
    print(f"   • Training all {total_params/1e9:.2f}B parameters")
    print(f"   • Target trajectory: {TARGET_TRAJECTORY}")
    print(f"   • Temperature: {INITIAL_TEMP} → {FINAL_TEMP}")
    print(f"   • Learning rate: {learning_rate} (base), {navigator_lr} (navigator)")
    print("="*60)
    
    global_step = 0
    best_val_accuracy = 0.0
    
    # Track metrics
    accuracy_by_length = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Anneal temperature
        temperature = INITIAL_TEMP - (INITIAL_TEMP - FINAL_TEMP) * (epoch / num_epochs)
        print(f"   Temperature: {temperature:.2f}")
        
        epoch_trajectories = []
        epoch_correct = []
        epoch_rewards = []
        epoch_losses = []
        epoch_natural_stops = []
        
        dataset_size = min(len(dataset), 1000)
        progress_bar = tqdm(range(dataset_size), desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for batch_idx in progress_bar:
            item = dataset[batch_idx % len(dataset)]
            question = item['question']
            answer_text = item['answer']
            
            train_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution: {answer_text}"
            inference_prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
            
            try:
                # Training forward pass
                inputs = tokenizer(
                    train_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding=False
                ).to(device)
                
                # VARIABLE target trajectory
                if epoch < 2:
                    # Early epochs: enforce some variation
                    target_traj = random.choice([3, 4, 5])
                else:
                    # Later epochs: let model choose more freely
                    target_traj = None
                
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids'],
                    return_trajectory=True,
                    target_trajectory=target_traj,
                    temperature=temperature
                )
                
                trajectory = outputs.get('trajectory', {})
                trajectory_length = len(trajectory.get('states', []))
                natural_stop = trajectory.get('natural_stop', False)
                
                epoch_trajectories.append(trajectory_length)
                epoch_natural_stops.append(natural_stop)
                
                # Check correctness for EVERY sample
                generated_answer = generate_simple(
                    model, tokenizer, inference_prompt,
                    max_new_tokens=100
                )
                
                pred_ans = parse_final_answer(generated_answer)
                true_ans = parse_final_answer(answer_text)
                
                correct = check_answer_correctness(generated_answer, answer_text)
                epoch_correct.append(correct)
                
                # Calculate reward with stopping bonus
                reward = compute_reward_v2(
                    trajectory_length, correct, natural_stop, 
                    target_length=TARGET_TRAJECTORY
                )
                epoch_rewards.append(reward)
                
                # Track accuracy by trajectory length
                accuracy_by_length[trajectory_length]['total'] += 1
                if correct:
                    accuracy_by_length[trajectory_length]['correct'] += 1
                
                # Debug output
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"\n  Sample {batch_idx}:")
                    print(f"  Q: {question[:80]}...")
                    print(f"  True: {true_ans}, Pred: {pred_ans}")
                    print(f"  Trajectory: {trajectory_length}, Natural stop: {natural_stop}")
                    print(f"  Correct: {correct}, Reward: {reward:.2f}")
                    if len(trajectory.get('continue_probs', [])) > 0:
                        probs = trajectory['continue_probs']
                        print(f"  Continue probs: {[f'{p:.2f}' for p in probs[:5]]}")
                
                # Compute loss with reward
                if outputs['loss'] is not None:
                    loss = outputs['loss']
                    
                    # Reward influences loss
                    loss = loss - 0.1 * reward
                    
                    # Add stopping penalty if always max length
                    if trajectory_length == MAX_TRAJECTORY:
                        loss = loss + 0.2  # Penalty for always maxing out
                    
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    epoch_losses.append(loss.item())
                
            except Exception as e:
                print(f"\n  ⚠️ Error in batch {batch_idx}: {e}")
                continue
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == dataset_size - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Update progress bar
            if len(epoch_correct) > 0:
                recent_acc = np.mean(epoch_correct[-50:]) if len(epoch_correct) >= 50 else np.mean(epoch_correct)
                recent_traj = np.mean(epoch_trajectories[-50:]) if len(epoch_trajectories) >= 50 else np.mean(epoch_trajectories)
                recent_reward = np.mean(epoch_rewards[-50:]) if len(epoch_rewards) >= 50 else np.mean(epoch_rewards)
                recent_stops = np.mean(epoch_natural_stops[-50:]) if len(epoch_natural_stops) >= 50 else np.mean(epoch_natural_stops)
                
                progress_bar.set_postfix({
                    'traj': f"{recent_traj:.1f}",
                    'acc': f"{recent_acc:.1%}",
                    'stop': f"{recent_stops:.1%}",
                    'reward': f"{recent_reward:.2f}"
                })
        
        # Epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  • Avg trajectory: {np.mean(epoch_trajectories):.2f}")
        print(f"  • Natural stops: {np.mean(epoch_natural_stops):.1%}")
        print(f"  • Accuracy: {np.mean(epoch_correct):.2%}")
        print(f"  • Avg reward: {np.mean(epoch_rewards):.3f}")
        
        # Print accuracy by trajectory length
        print(f"  • Accuracy by trajectory length:")
        for length in sorted(accuracy_by_length.keys()):
            stats = accuracy_by_length[length]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"    Length {length}: {acc:.2%} ({stats['total']} samples)")
        
        # Distribution of trajectories
        print(f"  • Trajectory distribution:")
        for i in range(MAX_TRAJECTORY + 1):
            count = epoch_trajectories.count(i)
            if count > 0:
                pct = count / len(epoch_trajectories) * 100
                bar = '█' * int(pct/2)
                print(f"    {i}: {bar:<25} {pct:.1f}%")
        
        # Validation
        print(f"\n🔍 Running validation...")
        val_correct = 0
        val_trajectories = []
        val_total = 30
        
        for i in tqdm(range(val_total), desc="Validating"):
            item = val_dataset[i]
            question = item['question']
            answer_text = item['answer']
            
            prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
            
            # Generate with natural stopping
            generated = generate_simple(model, tokenizer, prompt, max_new_tokens=100)
            
            if check_answer_correctness(generated, answer_text):
                val_correct += 1
        
        val_accuracy = val_correct / val_total
        print(f"📊 Validation Accuracy: {val_accuracy:.2%} ({val_correct}/{val_total})")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"🏆 New best validation accuracy!")
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': np.mean(epoch_correct),
                'avg_trajectory': np.mean(epoch_trajectories),
                'natural_stop_rate': np.mean(epoch_natural_stops)
            }, 'checkpoints/best_model_v5.pt')
        
        scheduler.step()
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.2%}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*generation flags.*")
    warnings.filterwarnings("ignore", message=".*requires_grad.*")
    
    train()
