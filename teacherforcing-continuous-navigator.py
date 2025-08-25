"""
COCONUT Training v31 - Definitive Tensor Integrity Fix
=====================================================
This script trains a reasoning-augmented language model. It definitively fixes
the "zero loss" bug by manually cloning tensors to prevent reference-based data
corruption, ensuring the Navigator receives a valid training signal.
Key Fixes:
- **Defensive Tensor Cloning**: A critical fix to prevent a subtle bug where
  predicted and ground-truth tensors were becoming corrupted.
- **Stable Training Loop**: Uses a stable and conceptually sound teacher forcing
  loop for reliable training.
- **Masked Cosine Loss**: Replaced MSE with a masked cosine similarity loss for a more
  stable and meaningful navigator training signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import os
import gc
import random
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
# Use notebook-friendly tqdm if in a Jupyter environment
if 'get_ipython' in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def generate_teacher_thoughts_mock(full_answer_text: str, tokenizer, device, dtype, embedding_layer, max_thoughts=5) -> Tuple[List[torch.Tensor], List[str]]:
    """Mocks a teacher model by extracting or creating reasoning steps."""
    main_text = full_answer_text.split('####')[0].strip()
    steps = re.split(r'\.\s+', main_text)
    thought_steps_text = [s.strip() for s in steps if len(s.split()) > 2 and len(s) > 10][:max_thoughts]
    
    if len(thought_steps_text) < 2:
        alt_steps = re.split(r'(?:Step \d+:|^\d+\.|^\d+\))', main_text)
        for step in alt_steps:
            step = step.strip()
            if len(step.split()) > 2 and step not in thought_steps_text:
                thought_steps_text.append(step)
                if len(thought_steps_text) >= max_thoughts: break
    
    if not thought_steps_text:
        final_answer = parse_final_answer(full_answer_text)
        if final_answer is not None:
            thought_steps_text = [f"Let me solve this step by step.", f"The final answer is {final_answer}."]
        else:
            thought_steps_text = ["Analyzing the problem.", "Computing the solution."]
    thought_embeds = []
    with torch.no_grad():
        for thought_text in thought_steps_text:
            thought_ids = tokenizer(thought_text, return_tensors='pt').input_ids.to(device)
            embed = embedding_layer(thought_ids).mean(dim=1).detach().clone()
            thought_embeds.append(embed)
            
    return thought_embeds, thought_steps_text
def parse_final_answer(text: str) -> Optional[float]:
    """Extracts the last numerical answer from a string."""
    if not text: return None
    text = text.replace(',', '')
    gsm_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if gsm_match: return float(gsm_match.group(1))
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers: return float(numbers[-1])
    return None
def check_answer_correctness(pred_text: str, true_text: str, tolerance: float = 1e-4) -> bool:
    """Compares the numerical answers from two strings."""
    pred_answer = parse_final_answer(pred_text)
    true_answer = parse_final_answer(true_text)
    if pred_answer is None or true_answer is None: return False
    return abs(pred_answer - true_answer) < tolerance
def plot_training_curves(train_losses, val_accuracies, lm_losses, nav_losses, is_smoke_test=False):
    """Visualizes the training and validation metrics."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    title_prefix = "Smoke Test" if is_smoke_test else "Training"
    ax1.plot(train_losses, label='Total Loss', color='blue'); ax1.plot(lm_losses, label='LM Loss', color='green', linestyle='--'); ax1.plot(nav_losses, label='Navigator Loss', color='orange', linestyle=':')
    ax1.set_title(f'{title_prefix} Loss Curves', fontsize=16); ax1.set_xlabel('Steps', fontsize=12); ax1.set_ylabel('Loss', fontsize=12); ax1.legend(); ax1.grid(True)
    x_axis_label = "Epoch (x5)" if not is_smoke_test else "Dummy Epoch"
    ax2.plot(val_accuracies, label='Validation Accuracy', color='purple', marker='o')
    ax2.set_title(f'{title_prefix} Validation Accuracy', fontsize=16); ax2.set_xlabel(x_axis_label, fontsize=12); ax2.set_ylabel('Accuracy', fontsize=12); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()
# ============================================================
# REASONING MODULES
# ============================================================
class ContinuousReasoningNavigator(nn.Module):
    def __init__(self, hidden_size, dropout_rate, device, dtype):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(device=device, dtype=dtype)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # FIX: Ensure state is a new tensor before operating
        state = state.clone()
        return state + self.ffn(state)
class TeacherForcingCoconut(nn.Module):
    def __init__(self, base_model: LlamaForCausalLM, dropout_rate: float):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.navigator = ContinuousReasoningNavigator(self.hidden_size, dropout_rate, device, dtype)
        # FIX: Gradient checkpointing is intentionally disabled to avoid conflicts
        # if hasattr(self.base_model, 'gradient_checkpointing_enable'):
        # self.base_model.gradient_checkpointing_enable()
    def forward(self, input_ids, attention_mask, prompt_lengths, teacher_thoughts_embeds):
        batch_size = input_ids.shape[0]; device = input_ids.device
        full_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        with torch.no_grad():
            prompt_mask = torch.zeros_like(attention_mask)
            for i, length in enumerate(prompt_lengths):
                prompt_mask[i, :length] = 1
            prompt_outputs = self.base_model(inputs_embeds=full_embeds, attention_mask=prompt_mask, output_hidden_states=True)
            initial_states = prompt_outputs.hidden_states[-1][torch.arange(batch_size), torch.tensor(prompt_lengths, device=device) - 1].detach().clone()
        max_num_thoughts = max(len(t) for t in teacher_thoughts_embeds)
        thought_loss = torch.tensor(0.0, device=device, dtype=full_embeds.dtype)
        
        if max_num_thoughts > 0:
            current_states = initial_states.clone()
            for i in range(max_num_thoughts):
                predicted_thoughts = self.navigator(current_states.clone())
                ground_truth_list = [t[i] if i < len(t) else torch.zeros_like(predicted_thoughts[j].unsqueeze(0)) for j, t in enumerate(teacher_thoughts_embeds)]
                ground_truth = torch.cat(ground_truth_list)
                
                # MASKED COSINE LOSS
                # 1. Create a mask to ignore padded (zero) ground-truth thoughts
                gt_norms = torch.norm(ground_truth, p=2, dim=-1)
                mask = gt_norms > 0
                
                # 2. Only compute loss where there's a valid ground-truth thought
                if mask.any():
                    cos_sim = F.cosine_similarity(predicted_thoughts[mask], ground_truth[mask], dim=-1)
                    # 3. The loss is 1 minus the cosine similarity, averaged over valid thoughts
                    step_loss = (1 - cos_sim).mean()
                    thought_loss += step_loss
                
                current_states = ground_truth.clone()
            thought_loss /= max_num_thoughts
        final_embeds_list, labels_list = [], []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            thoughts_tensor = torch.cat(teacher_thoughts_embeds[i]) if teacher_thoughts_embeds[i] else torch.empty(0, self.hidden_size, device=device, dtype=full_embeds.dtype)
            
            prompt_and_thoughts_embeds = torch.cat([full_embeds[i, :prompt_len], thoughts_tensor])
            answer_len = attention_mask[i].sum() - prompt_len
            answer_embeds = full_embeds[i, prompt_len : prompt_len + answer_len]
            answer_labels = input_ids[i, prompt_len : prompt_len + answer_len]
            
            final_embeds_list.append(torch.cat([prompt_and_thoughts_embeds, answer_embeds]))
            ignore_labels = torch.full((prompt_and_thoughts_embeds.shape[0],), -100, device=device, dtype=torch.long)
            labels_list.append(torch.cat([ignore_labels, answer_labels]))
        final_embeds = nn.utils.rnn.pad_sequence(final_embeds_list, batch_first=True)
        final_labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        final_mask = (final_embeds.sum(dim=-1) != 0).long()
        
        final_outputs = self.base_model(inputs_embeds=final_embeds, attention_mask=final_mask, labels=final_labels)
        
        total_loss = final_outputs.loss + 1.0 * thought_loss
        return {'loss': total_loss, 'lm_loss': final_outputs.loss.item(), 'thought_loss': thought_loss.item()}
    @torch.no_grad()
    def generate_with_reasoning_batched(self, tokenizer, prompt_texts: List[str], max_new_tokens=150, max_reasoning_steps=5):
        self.eval(); device = next(self.parameters()).device
        inputs = tokenizer(prompt_texts, return_tensors='pt', padding=True).to(device)
        prompt_embeds = self.base_model.get_input_embeddings()(inputs.input_ids)
        prompt_outputs = self.base_model(inputs_embeds=prompt_embeds, attention_mask=inputs.attention_mask, output_hidden_states=True)
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        current_states = prompt_outputs.hidden_states[-1][torch.arange(len(prompt_texts)), last_token_indices]
        
        thought_embeds_list = []
        for _ in range(max_reasoning_steps):
            latent_thoughts = self.navigator(current_states.clone())
            latent_thoughts_norm = F.normalize(latent_thoughts, p=2, dim=-1)
            thought_embeds_list.append(latent_thoughts_norm.unsqueeze(1))
            current_states = latent_thoughts_norm.clone()
            
        thoughts_tensor = torch.cat(thought_embeds_list, dim=1)
        combined_embeds = torch.cat([prompt_embeds, thoughts_tensor], dim=1)
        output_ids = self.base_model.generate(inputs_embeds=combined_embeds, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        self.train(); return responses
def quick_smoke_test(model, tokenizer, dataset, device, batch_size):
    """Runs a quick test to ensure the training and visualization are functional."""
    print("\n💨 Running a quick smoke test...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    test_train_losses, test_lm_losses, test_nav_losses = [], [], []
    for step in range(2):
        batch_items = [dataset[i] for i in range(step * batch_size, (step+1) * batch_size)]
        prompts = [f"Question: {item['question']}\n\nSolution:" for item in batch_items]
        full_texts = [f"{prompts[j]} {item['answer']}" for j, item in enumerate(batch_items)]
        prompt_lengths = [len(tokenizer.encode(p)) for p in prompts]
        full_inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        teacher_thoughts_batch_embeds = [generate_teacher_thoughts_mock(item['answer'], tokenizer, device, model.base_model.dtype, model.base_model.get_input_embeddings())[0] for item in batch_items]
        outputs = model(input_ids=full_inputs.input_ids, attention_mask=full_inputs.attention_mask, prompt_lengths=prompt_lengths, teacher_thoughts_embeds=teacher_thoughts_batch_embeds)
        loss = outputs['loss']
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        test_train_losses.append(outputs['loss'].detach().cpu()); test_lm_losses.append(outputs['lm_loss']); test_nav_losses.append(outputs['thought_loss'])
        print(f" Step {step+1}: Total Loss: {outputs['loss']:.4f}, LM Loss: {outputs['lm_loss']:.4f}, Nav Loss: {outputs['thought_loss']:.4f}")
    print("✅ Smoke test training loop passed!"); print("📊 Visualizing smoke test curves...")
    dummy_val_accuracies = [0.05, 0.10]
    plot_training_curves(test_train_losses, dummy_val_accuracies, test_lm_losses, test_nav_losses, is_smoke_test=True)
    print("✅ Smoke test visualization is functional. Metrics will now be reset for the main training run.")
def train():
    # --- Configuration ---
    num_epochs = 15; initial_batch_size = 2; target_effective_batch = 32; max_length = 512
    base_lr = 5e-6; navigator_lr = 2e-5; device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60 + "\nCOCONUT Training v31 - Definitive Tensor Integrity Fix\n" + "="*60)
    
    # --- Model Loading ---
    print("\n📚 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.bfloat16, device_map='auto')
    model = TeacherForcingCoconut(base_model=base_model, dropout_rate=0.1)
    
    model.to(device)
    
    smoke_test_dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
    quick_smoke_test(model, tokenizer, smoke_test_dataset, device, initial_batch_size)
    
    # --- Optimizer & Scheduler ---
    navigator_params = list(model.navigator.parameters())
    base_params_ids = set(p.data_ptr() for p in model.base_model.parameters())
    base_params = [p for p in model.parameters() if p.data_ptr() in base_params_ids]
    optimizer = torch.optim.AdamW([{'params': navigator_params, 'lr': navigator_lr}, {'params': base_params, 'lr': base_lr, 'weight_decay': 0.01}], betas=(0.9, 0.95))
    
    print("📊 Loading GSM8K dataset for full training...")
    dataset = smoke_test_dataset
    val_dataset = load_dataset("gsm8k", "main", split="test")
    
    num_training_steps = (min(len(dataset), 2000) // initial_batch_size) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
    # --- Training Loop ---
    print("\n🚀 Starting full training run...")
    batch_size = initial_batch_size
    best_val_accuracy = 0.0
    train_losses, lm_losses, nav_losses, val_accuracies = [], [], [], []
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        model.train()
        num_samples = min(len(dataset), 2000)
        progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}")
        gradient_accumulation_steps = max(1, target_effective_batch // batch_size)
        
        for step, i in enumerate(progress_bar):
            batch_items = [dataset[j] for j in range(i, min(i + batch_size, num_samples))]
            if not batch_items: continue
            prompts = [f"Question: {item['question']}\n\nSolution:" for item in batch_items]
            full_texts = [f"{prompts[j]} {item['answer']}" for j, item in enumerate(batch_items)]
            prompt_lengths = [len(tokenizer.encode(p)) for p in prompts]
            full_inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
            teacher_thoughts_batch_embeds = [generate_teacher_thoughts_mock(item['answer'], tokenizer, device, model.base_model.dtype, model.base_model.get_input_embeddings())[0] for item in batch_items]
            try:
                outputs = model(input_ids=full_inputs.input_ids, attention_mask=full_inputs.attention_mask, prompt_lengths=prompt_lengths, teacher_thoughts_embeds=teacher_thoughts_batch_embeds)
                loss = outputs['loss'] / gradient_accumulation_steps
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                    train_losses.append(outputs['loss'].detach().cpu()); lm_losses.append(outputs['lm_loss']); nav_losses.append(outputs['thought_loss'])
                progress_bar.set_postfix({'Loss': f"{outputs['loss']:.3f}", 'LM': f"{outputs['lm_loss']:.3f}", 'Nav': f"{outputs['thought_loss']:.3f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})
            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️ OOM Error! Skipping batch {step}."); gc.collect(); torch.cuda.empty_cache(); optimizer.zero_grad(); continue
            
        # --- Validation ---
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print(f"\n🔍 Running validation for Epoch {epoch+1}...")
            val_correct, val_total = 0, min(len(val_dataset), 100)
            val_progress_bar = tqdm(range(0, val_total, batch_size), desc="Validating")
            for i in val_progress_bar:
                val_batch_items = [val_dataset[j] for j in range(i, min(i + batch_size, val_total))]
                if not val_batch_items: continue
                prompts = [f"Question: {item['question']}\n\nSolution:" for item in val_batch_items]
                true_answers = [item['answer'] for item in val_batch_items]
                generated_answers = model.generate_with_reasoning_batched(tokenizer, prompts)
                for gen_ans, true_ans in zip(generated_answers, true_answers):
                    if check_answer_correctness(gen_ans, true_ans): val_correct += 1
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            val_accuracies.append(val_accuracy)
            print(f"\n📊 Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2%}")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"🏆 New best validation accuracy: {best_val_accuracy:.2%}")
            
    print("\n" + "="*60 + f"\n✅ Training complete! Best validation accuracy: {best_val_accuracy:.2%}\n" + "="*60)
    if train_losses and val_accuracies: plot_training_curves(train_losses, val_accuracies, lm_losses, nav_losses)
if __name__ == "__main__":
    train()
