"""
COCONUT Training v42.1 - Three-Phase Curriculum Learning (Corrected)
===================================================================
This version implements a structured, three-phase training regimen and fixes
critical bugs from the previous version, most notably the incorrect output
slicing during validation that caused 0% accuracy and failing smoke tests.

Key Changes from v42.1 (Original):
- **Corrected Reasoning Extraction**: The `extract_reasoning_steps` function
  was fixed to correctly parse sentences, allowing smoke tests to pass.
- **Robust Generation Logic**: The `generate_with_curriculum` method was
  overhauled to fix a tensor concatenation bug (`stack` vs. `cat`) and remove
  unreliable slicing of the output. The new logic assumes `generate` with
  `inputs_embeds` returns only new tokens and reconstructs the full response,
  preventing validation failures.
- **Reordered Training Strategy**: The training phases have been reordered to
  calibrate the navigator on the base model before joint fine-tuning.
  1. Phase 1 (Navigator Warm-up): Freezes the base LLM and trains only the
     navigator on CoT hidden states.
  2. Phase 2 (Joint CoT Fine-tuning): Unfreezes the LLM adapters and fine-tunes
     both components together on standard CoT data (Stage 0).
  3. Phase 3 (Latent Curriculum): Proceeds with the multi-stage COCONUT
     curriculum.
- **Heavy Regularization**: Increased dropout in the navigator and added weight
  decay to its optimizer to prevent overfitting.
- **Bug Fixes**: Corrected multiple minor bugs related to environment variables,
  tensor shapes, and list initialization.

USAGE:
------
# In a notebook:
main('test') # Run smoke tests only
main('train') # Run the full three-phase training
"""
import unsloth
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import os
import gc
import random
import re
import matplotlib.pyplot as plt
import unittest
from typing import List, Optional, Tuple
import bitsandbytes as bnb
from unittest.mock import Mock, patch

# Use notebook-friendly tqdm if in a Jupyter environment
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# --- Environment Setup ---
# This must be set before importing transformers
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("⚠️ Unsloth not available. Performance will be degraded.")
    FastLanguageModel = None
    UNSLOTH_AVAILABLE = False


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_reasoning_steps(answer_text: str) -> List[str]:
    """
    (CORRECTED) Extract individual reasoning steps from a GSM8K answer.
    This version uses a more robust regex to handle various sentence
    structures and correctly passes the smoke tests.
    """
    reasoning_part = answer_text.split('####')[0].strip()
    if not reasoning_part:
        return []

    # Normalize newlines and find all sequences of text that end with
    # sentence-terminating punctuation. This correctly handles multiple
    # sentences on one line and ignores text without punctuation, as
    # expected by the smoke tests.
    text_block = reasoning_part.replace('\n', ' ')
    sentences = re.findall(r'[^.!?]+[.!?]', text_block)

    return [s.strip() for s in sentences if s.strip()]


def prepare_data_for_stage(dataset_item, stage: int, c_thought: int, tokenizer, device, dtype, embedding_layer=None):
    """
    Prepare training data for a specific curriculum stage.
    Stage 0: Full CoT in language.
    Stage s > 0: Replace first s reasoning steps with c*s continuous thoughts.
    """
    question = dataset_item['question']
    answer_text = dataset_item['answer']

    reasoning_steps = extract_reasoning_steps(answer_text)
    final_answer_part = "####" + answer_text.split("####")[-1] if "####" in answer_text else ""

    if stage == 0:
        # Initial stage: Full CoT training
        prompt = f"Question: {question}\n\nSolution:"
        full_text = f"{prompt} {answer_text}"
        return prompt, full_text, [], reasoning_steps

    else:
        # Later stages: Replace first 'stage' steps with continuous thoughts
        num_steps_to_replace = min(stage, len(reasoning_steps))
        replaced_steps = reasoning_steps[:num_steps_to_replace]
        remaining_steps = reasoning_steps[num_steps_to_replace:]

        # Create the training text with placeholders for continuous thoughts
        prompt = f"Question: {question}\n\n<bot>Solution:"
        remaining_text = " ".join(remaining_steps) + " " + final_answer_part
        full_text = f"{prompt}<eot> {remaining_text.strip()}"

        # Generate embeddings for the replaced steps (teacher signals)
        continuous_thoughts = []
        if embedding_layer is not None and replaced_steps:
            with torch.no_grad():
                for step in replaced_steps:
                    for _ in range(c_thought):
                        try:
                            step_ids = tokenizer(step, return_tensors='pt')['input_ids'].to(device)
                            if step_ids.shape[1] == 0: continue # Skip empty steps
                            embed = embedding_layer(step_ids).mean(dim=1).detach()
                            continuous_thoughts.append(embed)
                        except Exception as e:
                            print(f"Warning: Could not embed step. {e}")
                            # Add a placeholder if embedding fails
                            embed = torch.randn(1, embedding_layer.weight.shape[1], device=device, dtype=dtype)
                            continuous_thoughts.append(embed)

        return prompt, full_text, continuous_thoughts, remaining_steps


def parse_final_answer(text: str) -> Optional[float]:
    """Extracts the final numerical answer from a string."""
    if not text: return None
    text = text.replace(',', '')
    gsm_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if gsm_match: return float(gsm_match.group(1))
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return float(numbers[-1]) if numbers else None

def check_answer_correctness(pred_text: str, true_text: str, tolerance: float = 1e-4) -> bool:
    """Compares the numerical answers from two strings."""
    pred_answer = parse_final_answer(pred_text)
    true_answer = parse_final_answer(true_text)
    if pred_answer is None or true_answer is None: return False
    return abs(pred_answer - true_answer) < tolerance

def plot_training_curves(train_losses, val_accuracies, stage_boundaries=None):
    """Plot training loss and validation accuracy curves."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training loss
    ax1.plot(train_losses, label='Training Loss', color='dodgerblue', alpha=0.8)
    if stage_boundaries:
        # Use a single label for all boundary lines
        added_label = False
        for boundary in stage_boundaries:
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.6, label='Phase Start' if not added_label else "")
            added_label = True
    ax1.set_title('Training Loss Across Phases', fontsize=14)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot validation accuracy
    ax2.plot(range(len(val_accuracies)), val_accuracies,
             label='Validation Accuracy', color='forestgreen', marker='o', linestyle='-')
    ax2.set_title('Validation Accuracy by Phase', fontsize=14)
    ax2.set_xlabel('Curriculum Phase/Stage', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xticks(range(len(val_accuracies)))
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


# ============================================================
# MODEL COMPONENTS
# ============================================================

class CurriculumGraphMemory:
    """Stores and manages continuous thought vectors."""
    def __init__(self, initial_state: torch.Tensor, max_thoughts: int = 20):
        self.nodes = [initial_state.clone().detach()]
        self.max_thoughts = max_thoughts

    def add_node(self, new_node: torch.Tensor):
        if len(self.nodes) < self.max_thoughts:
            self.nodes.append(new_node.clone().detach())

    def get_memory_state(self) -> torch.Tensor:
        return torch.stack(self.nodes)

    def __len__(self):
        return len(self.nodes)

class GraphAttentionNavigator(nn.Module):
    """
    Generates the next continuous thought by attending over memory.
    Includes heavy regularization (high dropout, LayerNorm).
    """
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size) # Added for stability
        )

    def forward(self, current_thought: torch.Tensor, memory: CurriculumGraphMemory) -> torch.Tensor:
        memory_state = memory.get_memory_state()
        # Shape for attention: (batch, seq_len, embed_dim)
        memory_nodes = memory_state.view(1, len(memory), self.hidden_size)
        query = current_thought.unsqueeze(0)

        context, _ = self.attention(query, memory_nodes, memory_nodes)
        fused_state = query + context
        next_thought = self.ffn(fused_state)
        return next_thought.squeeze(0)

class CurriculumCognitiveModel(nn.Module):
    """The main model, combining the base LLM with the memory and navigator."""
    def __init__(self, base_model, dropout_rate: float = 0.2):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Ensure navigator has the same dtype as the base model to prevent mismatches
        model_dtype = next(base_model.parameters()).dtype
        self.navigator = GraphAttentionNavigator(self.hidden_size, dropout_rate=dropout_rate).to(dtype=model_dtype)
        
        self.current_stage = 0
        self.c_thought = 2
        self.thought_loss_fct = nn.MSELoss() # Using MSELoss as suggested

    def set_curriculum_stage(self, stage: int):
        self.current_stage = stage

    def forward(self, input_ids, attention_mask, prompt_lengths, continuous_thoughts_embeds,
                  thought_loss_weight: float = 0.2):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.base_model.parameters()).dtype

        input_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Get initial hidden state after the prompt
        with torch.no_grad():
            prompt_mask = torch.zeros_like(attention_mask)
            for i, length in enumerate(prompt_lengths):
                prompt_mask[i, :length] = 1
            prompt_outputs = self.base_model(
                inputs_embeds=input_embeds, attention_mask=prompt_mask, output_hidden_states=True
            )
            initial_states = prompt_outputs.hidden_states[-1][
                torch.arange(batch_size), torch.tensor(prompt_lengths, device=device) - 1
            ].detach().clone()

        total_thought_loss = torch.tensor(0.0, device=device, dtype=dtype)
        final_embeds_list, labels_list = [], []

        for i in range(batch_size):
            num_teacher_thoughts = len(continuous_thoughts_embeds[i])
            if self.current_stage > 0 and num_teacher_thoughts > 0:
                init_state = initial_states[i].unsqueeze(0)
                memory = CurriculumGraphMemory(init_state)
                current_thought = memory.nodes[-1]
                predicted_thoughts = []
                for step in range(num_teacher_thoughts):
                    pred_thought = self.navigator(current_thought, memory)
                    predicted_thoughts.append(pred_thought)
                    # Always use teacher forcing for stability
                    teacher_thought = continuous_thoughts_embeds[i][step]
                    memory.add_node(teacher_thought)
                    current_thought = teacher_thought

                pred_tensor = torch.stack(predicted_thoughts)
                target_tensor = torch.cat(continuous_thoughts_embeds[i])
                total_thought_loss += self.thought_loss_fct(pred_tensor, target_tensor)
                thoughts_to_inject = torch.cat(memory.nodes[1:])
            else:
                thoughts_to_inject = torch.empty(0, self.hidden_size, device=device, dtype=dtype)

            # Construct final input for the LLM. The model will receive the full
            # context (prompt + thoughts + answer) but will only be trained to
            # predict the answer part.
            prompt_len = prompt_lengths[i]
            
            # The "answer" part of the embeddings and labels starts after the prompt.
            # This is not removing data, but rather separating context from the target.
            answer_embeds = input_embeds[i, prompt_len:]
            final_embeds = torch.cat([input_embeds[i, :prompt_len], thoughts_to_inject, answer_embeds])
            final_embeds_list.append(final_embeds)

            # Create corresponding labels. We set the label for prompt and thought
            # tokens to -100, which is the standard practice to ignore them in
            # the loss calculation. The model sees the full text but only learns
            # to predict the unmasked (answer) tokens.
            answer_labels = input_ids[i, prompt_len:]
            ignore_labels = torch.full((prompt_len + thoughts_to_inject.shape[0],), -100, device=device, dtype=torch.long)
            labels = torch.cat([ignore_labels, answer_labels])
            labels_list.append(labels)

        # Pad sequences for batch processing
        final_embeds = nn.utils.rnn.pad_sequence(final_embeds_list, batch_first=True, padding_value=0)
        final_labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        final_mask = (final_embeds.sum(dim=-1) != 0).long()

        # Final forward pass through the base model
        outputs = self.base_model(
            inputs_embeds=final_embeds, attention_mask=final_mask, labels=final_labels
        )

        # Combine losses
        lm_loss = outputs.loss
        # Note: Thought loss is *expected* to be 0.0 during Stage 0 (CoT fine-tuning)
        if self.current_stage > 0 and any(len(t) > 0 for t in continuous_thoughts_embeds):
            thought_loss = total_thought_loss / batch_size
            total_loss = lm_loss + thought_loss_weight * thought_loss
        else:
            thought_loss = torch.tensor(0.0, device=device)
            total_loss = lm_loss

        return {'loss': total_loss, 'lm_loss': lm_loss.item(), 'thought_loss': thought_loss.item()}

    def _get_cot_hidden_states(self, item, tokenizer) -> List[torch.Tensor]:
        """Extracts hidden states for each reasoning step in a CoT example."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        prompt, _, _, _ = prepare_data_for_stage(item, 0, self.c_thought, tokenizer, device, dtype)
        reasoning_steps = extract_reasoning_steps(item['answer'])
        if not reasoning_steps: return []

        states = []
        with torch.no_grad():
            # Initial state from prompt
            prompt_inputs = tokenizer(prompt, return_tensors='pt').to(device)
            prompt_out = self.base_model(**prompt_inputs, output_hidden_states=True)
            states.append(prompt_out.hidden_states[-1][0, -1, :].unsqueeze(0)) # [1, hidden]

            # States for each reasoning step
            for step in reasoning_steps:
                step_inputs = tokenizer(step, return_tensors='pt').to(device)
                if step_inputs.input_ids.shape[1] == 0: continue
                step_out = self.base_model(**step_inputs, output_hidden_states=True)
                step_hidden = step_out.hidden_states[-1].mean(dim=1) # [1, hidden]
                states.append(step_hidden)
        return states

    def generate_with_curriculum(self, tokenizer, prompt: str, max_new_tokens: int = 256,
                                 temperature: float = 0.1, top_p: float = 0.9):
        """
        CORRECTED: Generate text using the curriculum-aware model.
        This version fixes a tensor shape bug and uses a more reliable generation
        strategy that does not depend on slicing the output, which is brittle when
        using `inputs_embeds`.
        """
        self.eval()
        device = next(self.parameters()).device

        # Get initial hidden state from prompt
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        prompt_embeds = self.base_model.get_input_embeddings()(inputs.input_ids)
        with torch.no_grad():
            prompt_outputs = self.base_model(inputs_embeds=prompt_embeds, output_hidden_states=True)
            initial_state = prompt_outputs.hidden_states[-1][:, -1, :]

        # Generate continuous thoughts if in a latent stage
        memory = CurriculumGraphMemory(initial_state)
        num_thoughts = self.current_stage * self.c_thought
        thoughts_embeds = []
        if num_thoughts > 0:
            current_state = memory.nodes[-1]
            for _ in range(num_thoughts):
                with torch.no_grad():
                    next_thought = self.navigator(current_state, memory)
                thoughts_embeds.append(next_thought)
                memory.add_node(next_thought)
                current_state = next_thought

        # Combine prompt and thought embeddings
        if thoughts_embeds:
            # BUG FIX: Use torch.cat, not torch.stack. `thoughts_embeds` is a list
            # of tensors of shape (1, hidden_size). Stacking creates a wrong dimension.
            thoughts_tensor = torch.cat(thoughts_embeds, dim=0).unsqueeze(0)
            combined_embeds = torch.cat([prompt_embeds, thoughts_tensor], dim=1)
        else:
            combined_embeds = prompt_embeds

        # Use the model's generate function. When using `inputs_embeds` without
        # `input_ids`, the output typically contains *only* the new tokens. Slicing
        # is unreliable. We generate only new tokens and prepend the prompt manually.
        output_ids = self.base_model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.1), # Avoid temperature 0
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Reconstruct the full text for evaluation
        return prompt + " " + generated_text.strip()


# ============================================================
# TRAINING PHASE FUNCTIONS
# ============================================================

def run_navigator_validation(model, val_dataset, tokenizer, subset_size=50):
    """Runs validation for the navigator and returns average loss."""
    model.navigator.eval()
    total_val_loss = 0
    num_items = 0
    subset = val_dataset.select(range(min(subset_size, len(val_dataset))))
    loss_fct = nn.MSELoss()

    with torch.no_grad():
        for item in subset:
            states = model._get_cot_hidden_states(item, tokenizer)
            if len(states) < 2: continue

            memory = CurriculumGraphMemory(states[0])
            item_loss = 0
            for i in range(1, len(states)):
                pred = model.navigator(memory.nodes[-1], memory)
                loss = loss_fct(pred, states[i])
                item_loss += loss.item()
                memory.add_node(states[i]) # Use ground truth for next step

            total_val_loss += item_loss / (len(states) - 1)
            num_items += 1

    return total_val_loss / num_items if num_items > 0 else float('inf')


def run_navigator_warm_up(model, train_dataset, val_dataset, tokenizer, epochs, patience, lr, subset_size):
    """Phase 1: Train only the navigator on CoT states with early stopping."""
    print("\n" + "="*50 + "\n🛡️ PHASE 1: Navigator Warm-up\n" + "="*50)

    # Freeze base LLM, ensure navigator is trainable
    for param in model.base_model.parameters(): param.requires_grad = False
    for param in model.navigator.parameters(): param.requires_grad = True

    optimizer = torch.optim.Adam(model.navigator.parameters(), lr=lr, weight_decay=0.1)
    loss_fct = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_navigator_state = None

    subset = train_dataset.select(range(min(subset_size, len(train_dataset))))

    for epoch in range(epochs):
        model.navigator.train()
        epoch_loss = 0.0
        progress_bar = tqdm(subset, desc=f"Warm-up Epoch {epoch+1}/{epochs}")
        for item in progress_bar:
            states = model._get_cot_hidden_states(item, tokenizer)
            if len(states) < 2: continue

            memory = CurriculumGraphMemory(states[0])
            optimizer.zero_grad()
            total_item_loss = 0
            for i in range(1, len(states)):
                pred = model.navigator(memory.nodes[-1], memory)
                loss = loss_fct(pred, states[i])
                total_item_loss += loss
                memory.add_node(states[i]) # Teacher forcing

            if total_item_loss > 0:
                avg_loss = total_item_loss / (len(states) - 1)
                avg_loss.backward()
                optimizer.step()
                epoch_loss += avg_loss.item()
                progress_bar.set_postfix({'Loss': f"{avg_loss.item():.4f}"})

        # Validation for early stopping
        avg_val_loss = run_navigator_validation(model, val_dataset, tokenizer)
        print(f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_navigator_state = model.navigator.state_dict()
            print(f"🏆 New best validation loss: {best_val_loss:.4f}. Saving navigator state.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("🛑 Early stopping triggered. Navigator warm-up complete.")
            break

    if best_navigator_state:
        print("✅ Loaded best navigator weights.")
        model.navigator.load_state_dict(best_navigator_state)
    return epoch


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

def train_with_curriculum(skip_tests=False):
    """Main training function with the new three-phase curriculum."""
    if not UNSLOTH_AVAILABLE:
        print("\n❌ Error: Unsloth library is required.")
        return False
    if not skip_tests and not run_comprehensive_smoke_tests():
        print("\n❌ Smoke tests failed! Aborting training.")
        return False

    # --- Configuration ---
    config = {
        'c_thought': 2, 'max_latent_stages': 3,
        'warm_up_epochs': 10, 'warm_up_patience': 2, 'warm_up_lr': 5e-5,
        'cot_epochs': 3,
        'epochs_per_stage': 3,
        'batch_size': 8, 'max_length': 512,
        'base_lr': 1e-5, 'navigator_lr': 2e-5, 'navigator_weight_decay': 0.1,
        'thought_loss_weight': 0.2
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*60 + "\n🥥 COCONUT Training v42.1 - Three-Phase Curriculum\n" + "="*60)

    # --- Model & Tokenizer Loading ---
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<bot>', '<eot>']})

    base_model, _ = FastLanguageModel.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct', max_seq_length=config['max_length'],
        dtype=torch.bfloat16, load_in_4bit=True)
    base_model.resize_token_embeddings(len(tokenizer))

    base_model = FastLanguageModel.get_peft_model(
        base_model, r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none")

    model = CurriculumCognitiveModel(base_model=base_model).to(device)
    model.c_thought = config['c_thought']

    # --- Dataset Loading ---
    train_dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
    val_dataset = load_dataset("gsm8k", "main", split="test")

    # --- Training Loop Variables ---
    all_train_losses = []
    all_val_accuracies = []
    stage_boundaries = []

    # --- PHASE 1: NAVIGATOR WARM-UP ---
    print("\n" + "="*50 + "\n🛡️ PHASE 1: Navigator Warm-up\n" + "="*50)
    run_navigator_warm_up(
        model, train_dataset, val_dataset, tokenizer,
        epochs=config['warm_up_epochs'], patience=config['warm_up_patience'],
        lr=config['warm_up_lr'], subset_size=200)

    # --- Unfreeze base model for subsequent phases ---
    print("\nUnfreezing LLM adapters for joint training...")
    for param in model.base_model.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = True

    # --- PHASE 2: JOINT CoT FINE-TUNING (STAGE 0) ---
    print("\n" + "="*50 + "\n📚 PHASE 2: Joint CoT Fine-tuning\n" + "="*50)
    stage_boundaries.append(len(all_train_losses))
    
    stage = 0
    model.set_curriculum_stage(stage)
    epochs = config['cot_epochs']

    optimizer = bnb.optim.AdamW8bit([
        {'params': model.navigator.parameters(), 'lr': config['navigator_lr'], 'weight_decay': config['navigator_weight_decay']},
        {'params': [p for p in model.base_model.parameters() if p.requires_grad], 'lr': config['base_lr']}
    ])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        num_samples = min(len(train_dataset), 1000)
        progress_bar = tqdm(range(0, num_samples, config['batch_size']), desc=f"Stage {stage} Epoch {epoch+1}")

        for batch_start in progress_bar:
            batch_end = min(batch_start + config['batch_size'], num_samples)
            batch_items = [train_dataset[j] for j in range(batch_start, batch_end)]
            if not batch_items: continue

            embedding_layer = model.base_model.get_input_embeddings()
            model_dtype = next(model.parameters()).dtype
            prepared_data = [prepare_data_for_stage(item, stage, config['c_thought'], tokenizer, device, model_dtype, embedding_layer) for item in batch_items]
            prompts, full_texts, thoughts, _ = zip(*prepared_data)

            prompt_lengths = [len(tokenizer.encode(p)) for p in prompts]
            full_inputs = tokenizer(list(full_texts), return_tensors='pt', padding=True, truncation=True, max_length=config['max_length']).to(device)

            try:
                optimizer.zero_grad()
                outputs = model(full_inputs.input_ids, full_inputs.attention_mask, prompt_lengths, list(thoughts), thought_loss_weight=config['thought_loss_weight'])
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                all_train_losses.append(loss.item())
                progress_bar.set_postfix({'Loss': f"{loss.item():.3f}", 'LM': f"{outputs['lm_loss']:.2f}", 'Thought': f"{outputs['thought_loss']:.3f}"})
            except torch.cuda.OutOfMemoryError:
                print("OOM Error, skipping batch.")
                gc.collect(); torch.cuda.empty_cache()
                optimizer.zero_grad()

    # --- Validation for Phase 2 ---
    print(f"\n🔍 Validating at end of Phase 2 (CoT Fine-tuning)...")
    model.eval()
    val_correct, val_total = 0, min(len(val_dataset), 50)
    with torch.no_grad():
        for i in tqdm(range(val_total), desc=f"Validating Stage {stage}"):
            item = val_dataset[i]
            prompt = f"Question: {item['question']}\n\nSolution:"
            try:
                pred_text = model.generate_with_curriculum(tokenizer, prompt, max_new_tokens=256)
                if check_answer_correctness(pred_text, item['answer']):
                    val_correct += 1
            except Exception as e:
                print(f"Validation generation failed: {e}")
                continue
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    all_val_accuracies.append(val_accuracy)
    print(f"Phase 2 - Validation Accuracy: {val_accuracy:.2%} ({val_correct}/{val_total})")

    # --- PHASE 3: LATENT CURRICULUM (STAGES 1..N) ---
    print("\n" + "="*50 + "\n🧠 PHASE 3: Latent Curriculum Training\n" + "="*50)
    
    for stage in range(1, 1 + config['max_latent_stages']):
        phase_name = f"Latent Curriculum Stage {stage}"
        epochs = config['epochs_per_stage']
        print("\n" + "="*50 + f"\n {phase_name}\n" + "="*50)

        model.set_curriculum_stage(stage)
        stage_boundaries.append(len(all_train_losses))

        optimizer = bnb.optim.AdamW8bit([
            {'params': model.navigator.parameters(), 'lr': config['navigator_lr'], 'weight_decay': config['navigator_weight_decay']},
            {'params': [p for p in model.base_model.parameters() if p.requires_grad], 'lr': config['base_lr']}
        ])

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()
            num_samples = min(len(train_dataset), 1000)
            progress_bar = tqdm(range(0, num_samples, config['batch_size']), desc=f"Stage {stage} Epoch {epoch+1}")

            for batch_start in progress_bar:
                batch_end = min(batch_start + config['batch_size'], num_samples)
                batch_items = [train_dataset[j] for j in range(batch_start, batch_end)]
                if not batch_items: continue

                embedding_layer = model.base_model.get_input_embeddings()
                model_dtype = next(model.parameters()).dtype
                prepared_data = [prepare_data_for_stage(item, stage, config['c_thought'], tokenizer, device, model_dtype, embedding_layer) for item in batch_items]
                prompts, full_texts, thoughts, _ = zip(*prepared_data)

                prompt_lengths = [len(tokenizer.encode(p)) for p in prompts]
                full_inputs = tokenizer(list(full_texts), return_tensors='pt', padding=True, truncation=True, max_length=config['max_length']).to(device)

                try:
                    optimizer.zero_grad()
                    outputs = model(
                        full_inputs.input_ids, full_inputs.attention_mask,
                        prompt_lengths, list(thoughts),
                        thought_loss_weight=config['thought_loss_weight']
                    )
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()

                    all_train_losses.append(loss.item())
                    progress_bar.set_postfix({
                        'Loss': f"{loss.item():.3f}",
                        'LM': f"{outputs['lm_loss']:.2f}",
                        'Thought': f"{outputs['thought_loss']:.3f}"
                    })
                except torch.cuda.OutOfMemoryError:
                    print("OOM Error, skipping batch.")
                    gc.collect(); torch.cuda.empty_cache()
                    optimizer.zero_grad()

        print(f"\n🔍 Validating at end of Stage {stage}...")
        model.eval()
        val_correct, val_total = 0, min(len(val_dataset), 50)
        with torch.no_grad():
            for i in tqdm(range(val_total), desc=f"Validating Stage {stage}"):
                item = val_dataset[i]
                prompt = f"Question: {item['question']}\n\n{'<bot>' if stage > 0 else ''}Solution:"
                try:
                    pred_text = model.generate_with_curriculum(tokenizer, prompt, max_new_tokens=256)
                    if check_answer_correctness(pred_text, item['answer']):
                        val_correct += 1
                except Exception as e:
                    print(f"Validation generation failed: {e}")
                    continue

        val_accuracy = val_correct / val_total if val_total > 0 else 0
        all_val_accuracies.append(val_accuracy)
        print(f"Stage {stage} - Validation Accuracy: {val_accuracy:.2%} ({val_correct}/{val_total})")

    print("\n✅ Curriculum Training Complete!")
    if all_train_losses:
        plot_training_curves(all_train_losses, all_val_accuracies, stage_boundaries)
    return True


# ============================================================
# SMOKE TESTS & MAIN EXECUTION
# ============================================================

def run_comprehensive_smoke_tests():
    """A minimal set of tests to ensure core functions work."""
    print("\n" + "="*60 + "\n🔬 RUNNING COMPREHENSIVE SMOKE TESTS\n" + "="*60)
    try:
        # Test reasoning step extraction
        assert len(extract_reasoning_steps("Step 1. Step 2 is longer. #### 3")) == 2
        assert len(extract_reasoning_steps("Final answer is #### 4")) == 0
        assert len(extract_reasoning_steps("One step only. #### 5")) == 1
        print("✅ Test: extract_reasoning_steps passed.")

        # Test answer parsing
        assert parse_final_answer("The answer is #### 42.0") == 42.0
        assert abs(parse_final_answer("The answer is #### -1,234.5") - -1234.5) < 1e-6
        print("✅ Test: parse_final_answer passed.")

        # Test correctness checking
        assert check_answer_correctness("... #### 5.0", "... #### 5") == True
        assert check_answer_correctness("... #### 5.1", "... #### 5") == False
        print("✅ Test: check_answer_correctness passed.")

        print("\n✅ ALL SMOKE TESTS PASSED SUCCESSFULLY!")
        return True
    except AssertionError as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(run_mode='train'):
    """Main execution function."""
    if run_mode == 'test':
        run_comprehensive_smoke_tests()
    elif run_mode == 'train':
        train_with_curriculum(skip_tests=False)
    else:
        print(f"❌ Unknown run_mode: {run_mode}. Use 'test' or 'train'.")

if __name__ == "__main__":
    # In a script, you might parse sys.argv here.
    # For notebooks, calling main() directly is recommended.
    print("Running in interactive mode. Call main('test') or main('train').")
    # Example to run training:
    # main('train')

