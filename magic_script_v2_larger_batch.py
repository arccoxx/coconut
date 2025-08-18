"""
COCONUT Training v5 - FULL MODEL Training with Smart Auto-Scaling
Key features:
1. Training entire unfrozen network (all 32 layers)
2. Smart auto-scaling for optimal GPU utilization
3. Aggressive batch size adjustment based on actual memory usage
4. Dynamic learning rate scaling based on batch size
5. Reduce navigator bias (was too strong)
6. Add stopping rewards
7. Variable trajectory targets
8. Better reward shaping for different lengths
9. Stochastic trajectory sampling

Auto-Scaling Strategy:
- Starts with conservative batch_size=8
- Tests actual memory usage with single sample
- Auto-scales aggressively if utilization < 40%
- Moderate scaling if utilization 40-60%
- Conservative scaling if utilization 60-70%
- Auto-reduces if utilization > 92%

Memory Usage Estimates (bfloat16):
- Batch size 4: ~35-40GB (37-42% utilization)
- Batch size 8: ~65-70GB (69-74% utilization)
- Batch size 10: ~78-82GB (83-87% utilization) 
- Batch size 12: ~85-90GB (90-96% utilization)
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
from collections import defaultdict, Counter
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
    # Configuration for 94GB GPU with auto-scaling
    num_epochs = 5
    batch_size = 8  # Starting conservative, will auto-scale based on actual usage
    target_effective_batch = 32  # Target effective batch size
    gradient_accumulation_steps = 4  # Will be adjusted based on final batch_size
    max_batch_size = 16  # Maximum we'll auto-scale to
    min_batch_size = 4  # Minimum fallback size
    max_length = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Track OOM events and successful batches
    oom_count = 0
    total_successful_batches = 0
    
    # Base learning rates (will be scaled based on actual batch size)
    base_learning_rate = 5e-6
    base_navigator_lr = 1e-5
    
    # Memory monitoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Initial GPU memory: {initial_memory:.1f}GB")
    
    # Auto-Scaling Guide for 94GB GPU:
    # ========================================
    # Utilization < 40% (e.g., 33%): Aggressive scaling to suggested batch size
    # Utilization 40-60%: Moderate scaling (halfway to suggested)
    # Utilization 60-70%: Conservative scaling (+2 only)
    # Utilization 70-85%: No change (optimal range)
    # Utilization 85-92%: Warning but continue
    # Utilization > 92%: Auto-reduce to prevent OOM
    #
    # Expected batch sizes after auto-scaling:
    # - If starts at 8 with 33% util → scales to ~20 (but capped at 16)
    # - If starts at 8 with 50% util → scales to ~11
    # - If starts at 8 with 70% util → stays at 8
    
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
    
    # Quick memory test with first batch
    if torch.cuda.is_available():
        print("\n🔍 Testing memory usage with first batch...")
        try:
            test_item = dataset[0]
            test_prompt = f"Question: {test_item['question']}\nLet's solve this step by step.\n\nSolution: {test_item['answer']}"
            test_inputs = tokenizer(test_prompt, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            
            with torch.no_grad():
                _ = model(
                    input_ids=test_inputs['input_ids'],
                    attention_mask=test_inputs['attention_mask'],
                    labels=test_inputs['input_ids'],
                    return_trajectory=True,
                    temperature=INITIAL_TEMP
                )
            
            test_mem = torch.cuda.max_memory_allocated() / 1024**3
            estimated_batch_mem = test_mem * batch_size * 1.2  # 20% safety margin
            
            print(f"   Single sample: {test_mem:.1f}GB")
            print(f"   Estimated for batch={batch_size}: {estimated_batch_mem:.1f}GB")
            
            if estimated_batch_mem > 90:
                suggested = int(85 / test_mem)
                print(f"   ⚠️ May OOM! Auto-adjusting batch_size to {suggested}")
                batch_size = suggested
            elif estimated_batch_mem < 50:
                # Very underutilized - be aggressive
                suggested = min(max_batch_size, int(80 / test_mem))
                print(f"   💡 Very low utilization! Auto-adjusting batch_size to {suggested} for better GPU usage")
                batch_size = suggested
            elif estimated_batch_mem < 70:
                # Somewhat underutilized - moderate increase
                suggested = min(max_batch_size, int(75 / test_mem))
                print(f"   💡 Can increase batch_size to {suggested} for better utilization")
                # Auto-adjust if it's significantly underutilized
                if estimated_batch_mem < 60:
                    batch_size = suggested
                    print(f"   Auto-adjusting to batch_size={batch_size}")
                
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        except Exception as e:
            print(f"   Memory test failed: {e}")
    
    # Scale learning rates based on actual batch size
    # Rule: scale with sqrt of batch size increase from base of 4
    lr_scale = np.sqrt(batch_size / 4)
    learning_rate = base_learning_rate * lr_scale
    navigator_lr = base_navigator_lr * lr_scale
    
    # Adjust gradient accumulation to maintain target effective batch size
    gradient_accumulation_steps = max(1, target_effective_batch // batch_size)
    actual_effective_batch = batch_size * gradient_accumulation_steps
    
    print(f"\n📚 Configuration adjusted for batch_size={batch_size}:")
    print(f"   Gradient accumulation: {gradient_accumulation_steps} steps")
    print(f"   Effective batch size: {actual_effective_batch}")
    print(f"   Base LR: {learning_rate:.2e}, Navigator LR: {navigator_lr:.2e}")
    
    # Store the final batch size for use in training loops
    final_batch_size = batch_size
    
    print("="*60)
    
    # Optimizer with warmup - NOW we have learning rates defined
    navigator_params = list(model.navigator.parameters())
    navigator_param_ids = {id(p) for p in navigator_params}
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in navigator_param_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': navigator_params, 'lr': navigator_lr, 'weight_decay': 0.001, 'initial_lr': navigator_lr},
        {'params': base_params, 'lr': learning_rate, 'weight_decay': 0.01, 'initial_lr': learning_rate}
    ], betas=(0.9, 0.95))
    
    # Learning rate scheduler with warmup
    warmup_steps = 50  # Warmup for first 50 optimizer steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * (1000 // (batch_size * gradient_accumulation_steps)), eta_min=1e-6)
    
    # Load dataset
    print("📊 Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    val_dataset = load_dataset("gsm8k", "main", split="test")
    
    print("\n🚀 Starting FULL MODEL training with VARIABLE trajectories!")
    print(f"   • Training all {total_params/1e9:.2f}B parameters")
    print(f"   • Final batch size: {final_batch_size} × {gradient_accumulation_steps} accumulation = {actual_effective_batch} effective")
    print(f"   • Target trajectory: {TARGET_TRAJECTORY}")
    print(f"   • Temperature: {INITIAL_TEMP} → {FINAL_TEMP}")
    print(f"   • Learning rate: {learning_rate:.2e} (base), {navigator_lr:.2e} (navigator)")
    print("="*60)
    
    global_step = 0
    best_val_accuracy = 0.0
    
    # Track metrics
    accuracy_by_length = defaultdict(lambda: {'correct': 0, 'total': 0})
    trajectory_trends = []  # Track trajectory distribution over time
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Reset batch size if it was reduced in previous epoch
        if epoch > 0 and oom_count > 0:
            print(f"   Note: Batch size adjusted to {final_batch_size} due to previous OOM")
        
        # Anneal temperature
        temperature = INITIAL_TEMP - (INITIAL_TEMP - FINAL_TEMP) * (epoch / num_epochs)
        print(f"   Temperature: {temperature:.2f}")
        
        epoch_trajectories = []
        epoch_correct = []
        epoch_rewards = []
        epoch_losses = []
        epoch_natural_stops = []
        successful_batches = 0  # Track successful batches this epoch
        
        dataset_size = min(len(dataset), 1000)
        # Use final_batch_size for consistent batching
        num_batches = dataset_size // final_batch_size
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for batch_idx in progress_bar:
            # Dynamic batch size adjustment after OOM
            current_batch_size = final_batch_size
            if oom_count > 0 and final_batch_size > min_batch_size:
                # Reduce batch size after OOM
                current_batch_size = max(min_batch_size, final_batch_size - oom_count * 2)
                if batch_idx == 0:
                    print(f"\n⚠️ Adjusted batch size to {current_batch_size} due to {oom_count} OOM events")
            
            batch_items = []
            batch_questions = []
            batch_answers = []
            
            # Collect batch of samples (with current size)
            for i in range(current_batch_size):
                idx = (batch_idx * final_batch_size + i) % len(dataset)
                item = dataset[idx]
                batch_items.append(item)
                batch_questions.append(item['question'])
                batch_answers.append(item['answer'])
            
            # Process batch
            batch_correct = []
            batch_trajectory_lengths = []
            batch_natural_stops = []
            batch_rewards = []
            batch_losses = []
            
            for i in range(current_batch_size):
                question = batch_questions[i]
                answer_text = batch_answers[i]
                
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
                    
                    batch_trajectory_lengths.append(trajectory_length)
                    batch_natural_stops.append(natural_stop)
                    
                    # Check correctness for EVERY sample
                    generated_answer = generate_simple(
                        model, tokenizer, inference_prompt,
                        max_new_tokens=100
                    )
                    
                    pred_ans = parse_final_answer(generated_answer)
                    true_ans = parse_final_answer(answer_text)
                    
                    correct = check_answer_correctness(generated_answer, answer_text)
                    batch_correct.append(correct)
                    
                    # Calculate reward with stopping bonus
                    reward = compute_reward_v2(
                        trajectory_length, correct, natural_stop, 
                        target_length=TARGET_TRAJECTORY
                    )
                    batch_rewards.append(reward)
                    
                    # Track accuracy by trajectory length
                    accuracy_by_length[trajectory_length]['total'] += 1
                    if correct:
                        accuracy_by_length[trajectory_length]['correct'] += 1
                    
                    # Compute loss with reward
                    if outputs['loss'] is not None:
                        loss = outputs['loss']
                        
                        # Reward influences loss
                        loss = loss - 0.1 * reward
                        
                        # Add stopping penalty if always max length
                        if trajectory_length == MAX_TRAJECTORY:
                            loss = loss + 0.2  # Penalty for always maxing out
                        
                        # Scale by both batch size and gradient accumulation
                        loss = loss / (final_batch_size * gradient_accumulation_steps)
                        loss.backward()
                        batch_losses.append(loss.item())
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"\n  ⚠️ OOM in sample {i} of batch {batch_idx}!")
                    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f}GB")
                    oom_count += 1
                    
                    # Clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    
                    # Suggest reduction
                    if final_batch_size > min_batch_size:
                        suggested_size = max(min_batch_size, final_batch_size - 2)
                        print(f"  💡 Consider reducing batch_size to {suggested_size}")
                        final_batch_size = suggested_size  # Auto-reduce for next epoch
                    continue
                    
                except Exception as e:
                    print(f"\n  ⚠️ Error in sample {i} of batch {batch_idx}: {e}")
                    continue
            
            # Update epoch metrics
            epoch_trajectories.extend(batch_trajectory_lengths)
            epoch_correct.extend(batch_correct)
            epoch_rewards.extend(batch_rewards)
            epoch_losses.extend(batch_losses)
            epoch_natural_stops.extend(batch_natural_stops)
            
            # Track successful batch
            if len(batch_losses) > 0:
                successful_batches += 1
                total_successful_batches += 1
                
                # Auto-adjust batch size based on memory after first successful batch
                if epoch == 0 and successful_batches == 1 and torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                    util = (peak_mem / 94) * 100
                    
                    if util < 70:
                        # We have room to grow
                        suggested = min(max_batch_size, int(final_batch_size * (85 / util)))
                        print(f"\n💡 Memory utilization is only {util:.0f}%. Consider batch_size={suggested} for ~85% utilization")
                        
                        # Auto-increase more aggressively based on utilization
                        if util < 40:
                            # Very low utilization - increase to suggested value
                            new_batch_size = suggested
                            print(f"   Auto-increasing batch_size to {new_batch_size} (aggressive scaling)")
                        elif util < 60:
                            # Moderate utilization - increase halfway to suggested
                            new_batch_size = min(suggested, final_batch_size + (suggested - final_batch_size) // 2)
                            print(f"   Auto-increasing batch_size to {new_batch_size} (moderate scaling)")
                        else:
                            # Close to 70% - conservative increase
                            new_batch_size = min(suggested, final_batch_size + 2)
                            print(f"   Auto-increasing batch_size to {new_batch_size} (conservative scaling)")
                        
                        final_batch_size = new_batch_size
                        # Recalculate gradient accumulation for new batch size
                        gradient_accumulation_steps = max(1, target_effective_batch // final_batch_size)
                        print(f"   Gradient accumulation adjusted to {gradient_accumulation_steps} steps")
                        
                    elif util > 92:
                        # Too close to limit
                        suggested = max(min_batch_size, int(final_batch_size * (85 / util)))
                        print(f"\n⚠️ Memory utilization is {util:.0f}% (risky). Auto-reducing batch_size to {suggested} for safety")
                        final_batch_size = suggested
                        # Recalculate gradient accumulation for new batch size
                        gradient_accumulation_steps = max(1, target_effective_batch // final_batch_size)
            
            # Debug output with running averages, memory info, and trajectory distribution
            if batch_idx % 10 == 0 and batch_idx > 0:
                # Calculate running averages over last 50 samples (or all if less)
                window_size = min(50, len(epoch_correct))
                recent_correct = epoch_correct[-window_size:] if window_size > 0 else []
                recent_trajectories = epoch_trajectories[-window_size:] if window_size > 0 else []
                recent_stops = epoch_natural_stops[-window_size:] if window_size > 0 else []
                recent_rewards = epoch_rewards[-window_size:] if window_size > 0 else []
                
                # Memory stats
                if torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / 1024**3
                    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                    utilization = (peak_mem / 94) * 100
                
                print(f"\n  📊 Batch {batch_idx} (samples {batch_idx * final_batch_size}-{(batch_idx + 1) * final_batch_size - 1}):")
                print(f"     Current batch (n={current_batch_size}): {np.mean(batch_correct):.0%} correct")
                print(f"     Last 50 samples: {np.mean(recent_correct):.1%} correct")
                print(f"     Epoch so far: {np.mean(epoch_correct):.1%} correct")
                
                # Trajectory statistics
                print(f"\n     🎯 Trajectory Stats:")
                print(f"        Current batch avg: {np.mean(batch_trajectory_lengths):.1f} steps")
                print(f"        Last 50 samples avg: {np.mean(recent_trajectories):.1f} steps")
                print(f"        Epoch avg: {np.mean(epoch_trajectories):.1f} steps")
                
                # Trajectory distribution for current batch
                traj_dist = {}
                for traj_len in batch_trajectory_lengths:
                    traj_dist[traj_len] = traj_dist.get(traj_len, 0) + 1
                
                print(f"        Distribution (current batch):")
                for steps in sorted(traj_dist.keys()):
                    count = traj_dist[steps]
                    pct = (count / len(batch_trajectory_lengths)) * 100
                    bar = '█' * int(pct / 5)  # Scale to fit
                    print(f"          {steps} steps: {bar:<20} {count}/{len(batch_trajectory_lengths)} ({pct:.0f}%)")
                
                # Trajectory vs correctness correlation
                correct_by_traj = {}
                for i, traj_len in enumerate(batch_trajectory_lengths):
                    if traj_len not in correct_by_traj:
                        correct_by_traj[traj_len] = {'correct': 0, 'total': 0}
                    correct_by_traj[traj_len]['total'] += 1
                    if batch_correct[i]:
                        correct_by_traj[traj_len]['correct'] += 1
                
                print(f"        Accuracy by trajectory length (batch):")
                for steps in sorted(correct_by_traj.keys()):
                    stats = correct_by_traj[steps]
                    acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                    print(f"          {steps} steps: {acc:.0f}% correct ({stats['correct']}/{stats['total']})")
                
                print(f"\n     Natural stops: {np.mean(recent_stops):.1%}")
                print(f"     Avg reward: {np.mean(recent_rewards):.2f}")
                
                if torch.cuda.is_available():
                    print(f"     💾 Memory: {current_mem:.1f}GB current, {peak_mem:.1f}GB peak ({utilization:.0f}% util)")
                
                # Show sample question with trajectory info
                if len(batch_questions) > 0 and len(batch_trajectory_lengths) > 0:
                    sample_idx = random.randint(0, min(len(batch_questions), len(batch_trajectory_lengths)) - 1)
                    sample_q = batch_questions[sample_idx]
                    sample_traj = batch_trajectory_lengths[sample_idx]
                    sample_correct = batch_correct[sample_idx] if sample_idx < len(batch_correct) else False
                    
                    print(f"\n     📝 Sample from batch:")
                    print(f"        Question: '{sample_q[:100]}{'...' if len(sample_q) > 100 else ''}'")
                    print(f"        Used {sample_traj} reasoning steps → {'✓' if sample_correct else '✗'} correct")
                    print(f"        Question complexity: {len(sample_q.split())} words")
                
                # Track trajectory trends every 50 batches
                if batch_idx % 50 == 0:
                    trend_snapshot = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'distribution': dict(traj_dist),
                        'avg_steps': np.mean(recent_trajectories),
                        'accuracy': np.mean(recent_correct)
                    }
                    trajectory_trends.append(trend_snapshot)
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                # Learning rate warmup
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['initial_lr'] * lr_scale if 'initial_lr' in param_group else learning_rate * lr_scale
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Step scheduler after warmup
                if global_step >= warmup_steps:
                    scheduler.step()
                
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Update progress bar with running averages including trajectory details
            if len(epoch_correct) > 0:
                # Use larger windows for more stable metrics
                window_size = min(100, len(epoch_correct))
                recent_acc = np.mean(epoch_correct[-window_size:]) if len(epoch_correct) >= window_size else np.mean(epoch_correct)
                recent_traj = np.mean(epoch_trajectories[-window_size:]) if len(epoch_trajectories) >= window_size else np.mean(epoch_trajectories)
                recent_reward = np.mean(epoch_rewards[-window_size:]) if len(epoch_rewards) >= window_size else np.mean(epoch_rewards)
                recent_stops = np.mean(epoch_natural_stops[-window_size:]) if len(epoch_natural_stops) >= window_size else np.mean(epoch_natural_stops)
                
                # Calculate trajectory mode (most common length)
                if len(epoch_trajectories) > 0:
                    traj_mode = Counter(epoch_trajectories[-window_size:]).most_common(1)[0][0]
                else:
                    traj_mode = 0
                
                postfix_dict = {
                    'steps': f"{recent_traj:.1f}",
                    'mode': f"{traj_mode}",
                    'acc': f"{recent_acc:.1%}",
                    'stop': f"{recent_stops:.1%}",
                    'reward': f"{recent_reward:.2f}",
                    'total': f"{np.mean(epoch_correct):.1%}"
                }
                
                # Add memory info every 20 batches
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / 1024**3
                    postfix_dict['mem'] = f"{mem_gb:.1f}GB"
                
                progress_bar.set_postfix(postfix_dict)
        
        # Epoch summary with detailed trajectory analysis
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  • Accuracy: {np.mean(epoch_correct):.2%}")
        print(f"  • Avg reward: {np.mean(epoch_rewards):.3f}")
        
        # Detailed trajectory statistics
        print(f"\n  🎯 Trajectory Analysis:")
        print(f"  • Average trajectory: {np.mean(epoch_trajectories):.2f} steps")
        print(f"  • Std deviation: {np.std(epoch_trajectories):.2f}")
        print(f"  • Min/Max: {min(epoch_trajectories) if epoch_trajectories else 0}/{max(epoch_trajectories) if epoch_trajectories else 0}")
        print(f"  • Natural stops: {np.mean(epoch_natural_stops):.1%}")
        
        # Overall trajectory distribution
        print(f"\n  • Trajectory distribution (full epoch):")
        epoch_traj_dist = {}
        for traj_len in epoch_trajectories:
            epoch_traj_dist[traj_len] = epoch_traj_dist.get(traj_len, 0) + 1
        
        for i in range(MAX_TRAJECTORY + 1):
            count = epoch_traj_dist.get(i, 0)
            if count > 0:
                pct = count / len(epoch_trajectories) * 100
                bar = '█' * int(pct/2)
                print(f"    {i} steps: {bar:<25} {count} samples ({pct:.1f}%)")
        
        # Accuracy by trajectory length (full epoch)
        print(f"\n  • Accuracy by trajectory length:")
        for length in sorted(accuracy_by_length.keys()):
            stats = accuracy_by_length[length]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                pct_of_total = (stats['total'] / len(epoch_trajectories)) * 100 if epoch_trajectories else 0
                print(f"    {length} steps: {acc:.2%} accuracy ({stats['total']} samples, {pct_of_total:.1f}% of epoch)")
        
        # Find optimal trajectory length
        best_acc = 0
        best_length = 0
        for length, stats in accuracy_by_length.items():
            if stats['total'] >= 10:  # Only consider lengths with enough samples
                acc = stats['correct'] / stats['total']
                if acc > best_acc:
                    best_acc = acc
                    best_length = length
        
        if best_length > 0:
            print(f"\n  📌 Best performing trajectory: {best_length} steps with {best_acc:.2%} accuracy")
        
        # Validation (increased samples for better metrics)
        print(f"\n🔍 Running validation...")
        val_correct = 0
        val_trajectories = []
        val_trajectory_lengths = []
        val_total = 100  # Increased from 30 for more stable metrics
        
        for i in tqdm(range(val_total), desc="Validating"):
            item = val_dataset[i]
            question = item['question']
            answer_text = item['answer']
            
            prompt = f"Question: {question}\nLet's solve this step by step.\n\nSolution:"
            
            # Get trajectory info during validation
            try:
                val_inputs = tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                with torch.no_grad():
                    val_outputs = model(
                        input_ids=val_inputs['input_ids'],
                        attention_mask=val_inputs['attention_mask'],
                        return_trajectory=True,
                        temperature=0.5  # Lower temperature for validation
                    )
                
                val_trajectory = val_outputs.get('trajectory', {})
                val_traj_length = len(val_trajectory.get('states', []))
                val_trajectory_lengths.append(val_traj_length)
                
                # Generate answer
                generated = generate_simple(model, tokenizer, prompt, max_new_tokens=100)
                
                if check_answer_correctness(generated, answer_text):
                    val_correct += 1
                    
            except Exception as e:
                print(f"\n  Validation error on sample {i}: {e}")
                val_trajectory_lengths.append(0)
                continue
        
        val_accuracy = val_correct / val_total
        avg_val_trajectory = np.mean(val_trajectory_lengths) if val_trajectory_lengths else 0
        
        print(f"\n📊 Validation Results:")
        print(f"  • Accuracy: {val_accuracy:.2%} ({val_correct}/{val_total})")
        print(f"  • Avg trajectory: {avg_val_trajectory:.2f} steps")
        
        # Validation trajectory distribution
        if val_trajectory_lengths:
            val_traj_dist = {}
            for traj_len in val_trajectory_lengths:
                val_traj_dist[traj_len] = val_traj_dist.get(traj_len, 0) + 1
            
            print(f"  • Trajectory distribution (validation):")
            for steps in sorted(val_traj_dist.keys()):
                count = val_traj_dist[steps]
                pct = (count / len(val_trajectory_lengths)) * 100
                bar = '█' * int(pct/3)
                print(f"    {steps} steps: {bar:<20} {count} ({pct:.0f}%)")
        
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
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"📈 Total successful batches: {total_successful_batches}")
    
    # Trajectory evolution analysis
    if trajectory_trends:
        print(f"\n🎯 Trajectory Evolution During Training:")
        print("  How average steps changed over time:")
        for i, trend in enumerate(trajectory_trends[-5:]):  # Show last 5 snapshots
            print(f"  Epoch {trend['epoch']+1}, Batch {trend['batch']}: {trend['avg_steps']:.1f} avg steps, {trend['accuracy']:.1%} acc")
        
        # Check if model converged to specific trajectory length
        final_trends = trajectory_trends[-3:] if len(trajectory_trends) >= 3 else trajectory_trends
        avg_steps_variance = np.std([t['avg_steps'] for t in final_trends])
        if avg_steps_variance < 0.5:
            final_avg = np.mean([t['avg_steps'] for t in final_trends])
            print(f"\n  ✓ Model converged to ~{final_avg:.1f} reasoning steps")
        else:
            print(f"\n  ✓ Model maintains diverse trajectory lengths (std: {avg_steps_variance:.2f})")
    
    if torch.cuda.is_available():
        final_peak = torch.cuda.max_memory_allocated() / 1024**3
        final_util = (final_peak / 94) * 100
        print(f"\n💾 Peak memory usage: {final_peak:.1f}GB ({final_util:.0f}% utilization)")
        
        if final_util < 80:
            suggested_final = int(final_batch_size * (85 / final_util))
            print(f"💡 For future runs, consider batch_size={suggested_final} for optimal GPU usage")
        
    if oom_count > 0:
        print(f"⚠️ Encountered {oom_count} OOM errors. Final batch_size was adjusted to {final_batch_size}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*generation flags.*")
    warnings.filterwarnings("ignore", message=".*requires_grad.*")
    
    train()
