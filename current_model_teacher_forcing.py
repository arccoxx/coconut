"""
COCONUT Training v39 - Comprehensive Unit Tests
=====================================================
This script trains a reasoning-augmented language model. This version replaces
the basic smoke tests with a comprehensive unit testing suite using Python's
`unittest` framework to validate every component—including evaluation and
graphing—before training begins.

Key Features:
- **Comprehensive Unit Tests**: A formal `unittest` suite validates utility
  functions, GraphMemory, the Navigator, the full training step, the evaluation
  step, and the final graphing function.
- **Unsloth Integration**: Uses FastLanguageModel for up to 2x faster training
  and 60% less memory usage, with 4-bit quantization enabled.
- **High-Quality Teacher Signal**: Generates structured, logical reasoning steps.
- **Balanced Loss Weighting**: Prioritizes final answer correctness.
- **PEFT (LoRA) & 8-bit Optimizer**: Ensures stable and efficient fine-tuning.
"""
import unsloth
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
import traceback
import unittest
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from peft import LoraConfig
import bitsandbytes as bnb
from unsloth import FastLanguageModel


# Use notebook-friendly tqdm if in a Jupyter environment
if 'get_ipython' in globals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def generate_structured_teacher_thoughts(question: str, full_answer_text: str, tokenizer, device, dtype, embedding_layer, max_thoughts=5) -> Tuple[List[torch.Tensor], List[str]]:
    """Generates a more structured and logical set of teacher thoughts."""
    thought_steps_text = []
    thought_steps_text.append("Let's break down the problem to identify the key steps.")
    question_numbers = re.findall(r'[-+]?\d*\.?\d+', question.replace(',', ''))
    if question_numbers:
        unique_numbers = sorted(list(set(float(n) for n in question_numbers)))
        thought_steps_text.append(f"The key numbers in the problem are {unique_numbers}.")
    final_answer = parse_final_answer(full_answer_text)
    if final_answer is not None:
        thought_steps_text.append(f"The goal is to calculate a final value, which should be {final_answer}.")
    thought_steps_text.append("By combining these pieces of information, I can derive the solution.")
    if len(thought_steps_text) < 3:
        thought_steps_text.extend(["I need to identify the core operations required.", "Executing the steps in order is crucial."])
    thought_steps_text = thought_steps_text[:max_thoughts]
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
    title_prefix = "Unit Test" if is_smoke_test else "Training"
    ax1.plot(train_losses, label='Total Loss', color='blue'); ax1.plot(lm_losses, label='LM Loss', color='green', linestyle='--'); ax1.plot(nav_losses, label='Navigator Loss', color='orange', linestyle=':')
    ax1.set_title(f'{title_prefix} Loss Curves', fontsize=16); ax1.set_xlabel('Steps', fontsize=12); ax1.set_ylabel('Loss', fontsize=12); ax1.legend(); ax1.grid(True)
    x_axis_label = "Epoch (x5)" if not is_smoke_test else "Dummy Epoch"
    ax2.plot(val_accuracies, label='Validation Accuracy', color='purple', marker='o')
    ax2.set_title(f'{title_prefix} Validation Accuracy', fontsize=16); ax2.set_xlabel(x_axis_label, fontsize=12); ax2.set_ylabel('Accuracy', fontsize=12); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    if is_smoke_test:
        plt.close(fig)
    else:
        plt.show()

# ============================================================
# DYNAMIC GRAPH MEMORY COMPONENTS
# ============================================================
class GraphMemory:
    """Manages a dynamic graph of thought nodes for a single reasoning process."""
    def __init__(self, initial_state: torch.Tensor):
        self.nodes = [initial_state.clone()]
        self.device = initial_state.device
        self.dtype = initial_state.dtype

    def add_node(self, new_node: torch.Tensor):
        self.nodes.append(new_node.clone())

    def get_memory_state(self) -> torch.Tensor:
        return torch.stack(self.nodes)

    def __len__(self):
        return len(self.nodes)

class GraphAttentionNavigator(nn.Module):
    """A navigator that uses graph attention to interact with a working memory."""
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_rate, batch_first=True).to(device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(device=device, dtype=dtype)

    def forward(self, current_thought: torch.Tensor, memory: GraphMemory) -> torch.Tensor:
        memory_state = memory.get_memory_state()
        num_nodes = memory_state.shape[0]
        memory_nodes = memory_state.view(1, num_nodes, self.hidden_size) 
        query = current_thought.unsqueeze(0) 
        context, _ = self.attention(query, memory_nodes, memory_nodes)
        fused_state = query + context
        next_thought = self.ffn(fused_state)
        return next_thought.squeeze(0)

class CognitiveGraphModel(nn.Module):
    """The main model integrating the LLM with the graph memory and navigator."""
    def __init__(self, base_model: LlamaForCausalLM, dropout_rate: float):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.navigator = GraphAttentionNavigator(self.hidden_size, dropout_rate=dropout_rate)

    def forward(self, input_ids, attention_mask, prompt_lengths, teacher_thoughts_embeds, thought_loss_weight: float = 1.0, epsilon: float = 1.0):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.base_model.parameters()).dtype
        
        full_embeds = self.base_model.get_input_embeddings()(input_ids)

        with torch.no_grad():
            prompt_mask = torch.zeros_like(attention_mask)
            for i, length in enumerate(prompt_lengths):
                prompt_mask[i, :length] = 1
            prompt_outputs = self.base_model(inputs_embeds=full_embeds, attention_mask=prompt_mask, output_hidden_states=True)
            initial_states = prompt_outputs.hidden_states[-1][torch.arange(batch_size), torch.tensor(prompt_lengths, device=device) - 1].detach().clone()

        max_num_thoughts = max(len(t) for t in teacher_thoughts_embeds)
        total_thought_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        final_embeds_list, labels_list = [], []
        for i in range(batch_size):
            memory = GraphMemory(initial_states[i].unsqueeze(0))
            current_navigator_state = memory.nodes[-1]
            predicted_thoughts_for_item = []
            
            for step in range(max_num_thoughts):
                predicted_thought = self.navigator(current_navigator_state, memory)
                predicted_thought = F.normalize(predicted_thought, p=2, dim=-1)
                predicted_thoughts_for_item.append(predicted_thought)

                use_teacher_forcing = random.random() < epsilon
                has_ground_truth = step < len(teacher_thoughts_embeds[i])

                if use_teacher_forcing and has_ground_truth:
                    next_input_thought = teacher_thoughts_embeds[i][step]
                else:
                    next_input_thought = predicted_thought
                
                memory.add_node(next_input_thought)
                current_navigator_state = next_input_thought
            
            if predicted_thoughts_for_item:
                predicted_thoughts_tensor = torch.cat(predicted_thoughts_for_item)
                ground_truth_tensor = torch.cat(teacher_thoughts_embeds[i])
                
                len_pred, len_gt = predicted_thoughts_tensor.shape[0], ground_truth_tensor.shape[0]
                if len_pred > len_gt: predicted_thoughts_tensor = predicted_thoughts_tensor[:len_gt]
                elif len_gt > len_pred:
                    padding = torch.zeros(len_gt - len_pred, self.hidden_size, device=device, dtype=dtype)
                    predicted_thoughts_tensor = torch.cat([predicted_thoughts_tensor, padding])

                cos_sim = F.cosine_similarity(predicted_thoughts_tensor, ground_truth_tensor, dim=-1)
                item_loss = (1 - cos_sim).mean()
                total_thought_loss += item_loss

            prompt_len = prompt_lengths[i]
            thoughts_for_lm = memory.nodes[1:]
            thoughts_tensor = torch.cat(thoughts_for_lm) if thoughts_for_lm else torch.empty(0, self.hidden_size, device=device, dtype=dtype)
            
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
        
        thought_loss = total_thought_loss / batch_size
        
        total_loss = final_outputs.loss + thought_loss_weight * thought_loss
        return {'loss': total_loss, 'lm_loss': final_outputs.loss.item(), 'thought_loss': thought_loss.item()}

    @torch.no_grad()
    def generate_with_reasoning_batched(self, tokenizer, prompt_texts: List[str], max_new_tokens=150, max_reasoning_steps=5):
        self.eval()
        device = next(self.parameters()).device
        
        inputs = tokenizer(prompt_texts, return_tensors='pt', padding=True).to(device)
        prompt_embeds = self.base_model.get_input_embeddings()(inputs.input_ids)
        
        prompt_outputs = self.base_model(inputs_embeds=prompt_embeds, attention_mask=inputs.attention_mask, output_hidden_states=True)
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1
        initial_states = prompt_outputs.hidden_states[-1][torch.arange(len(prompt_texts)), last_token_indices]

        final_combined_embeds = []
        for i in range(len(prompt_texts)):
            memory = GraphMemory(initial_states[i].unsqueeze(0))
            current_navigator_state = memory.nodes[-1]
            
            for _ in range(max_reasoning_steps):
                next_thought = self.navigator(current_navigator_state, memory)
                next_thought = F.normalize(next_thought, p=2, dim=-1)
                memory.add_node(next_thought)
                current_navigator_state = next_thought
            
            thoughts_tensor = torch.cat(memory.nodes[1:])
            prompt_len = inputs.attention_mask[i].sum()
            final_combined_embeds.append(torch.cat([prompt_embeds[i, :prompt_len], thoughts_tensor]))

        padded_embeds = nn.utils.rnn.pad_sequence(final_combined_embeds, batch_first=True)
        attention_mask = (padded_embeds.sum(dim=-1) != 0).long()

        # --- Manual Generation Loop FIX ---
        outputs = self.base_model(inputs_embeds=padded_embeds, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated_tokens = [next_tokens]
        batch_size = padded_embeds.shape[0]
        eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id], device=device)
        unfinished_sequences = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        past_length = past_key_values[0][0].shape[2]

        for _ in range(max_new_tokens - 1):
            position_ids = torch.tensor([[past_length]], device=device)
            
            model_inputs = self.base_model.prepare_inputs_for_generation(
                next_tokens, 
                past_key_values=past_key_values, 
                attention_mask=attention_mask
            )
            model_inputs['position_ids'] = position_ids

            outputs = self.base_model(**model_inputs, return_dict=True, use_cache=True)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_tokens.append(next_tokens)
            past_key_values = outputs.past_key_values
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long)], dim=1)
            past_length += 1

            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id_tensor).long())
            if unfinished_sequences.max() == 0:
                break

        all_generated_ids = torch.cat(generated_tokens, dim=1)
        
        responses = []
        for i in range(batch_size):
            generated_text = tokenizer.decode(all_generated_ids[i], skip_special_tokens=True)
            full_response = prompt_texts[i] + " " + generated_text
            responses.append(full_response)
            
        self.train()
        return responses

# ============================================================
# UNIT TESTS
# ============================================================
class TestCognitiveArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model and tokenizer once for all tests."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16
        cls.hidden_size = 4096 # Llama-3-8B
        cls.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name='meta-llama/Meta-Llama-3-8B-Instruct',
            max_seq_length=512,
            dtype=cls.dtype,
            load_in_4bit=True,
        )
        base_model = FastLanguageModel.get_peft_model(
            base_model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        
        cls.model = CognitiveGraphModel(base_model=base_model, dropout_rate=0.1).to(cls.device)
        cls.dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)

    def test_01_utility_functions(self):
        self.assertEqual(parse_final_answer("The answer is #### 1,234.5"), 1234.5)
        self.assertEqual(parse_final_answer("it is 42."), 42.0)
        self.assertTrue(check_answer_correctness("Final answer: 10", "#### 10"))
        self.assertFalse(check_answer_correctness("Final answer: 10.1", "#### 10"))

    def test_02_graph_memory(self):
        initial_state = torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype)
        memory = GraphMemory(initial_state)
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory.nodes[0].shape, (1, self.hidden_size))
        
        new_node = torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype)
        memory.add_node(new_node)
        self.assertEqual(len(memory), 2)
        
        state_tensor = memory.get_memory_state()
        self.assertEqual(state_tensor.shape, (2, 1, self.hidden_size))

    def test_03_navigator(self):
        navigator = self.model.navigator
        initial_state = torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype)
        memory = GraphMemory(initial_state)
        
        current_thought = memory.nodes[-1]
        next_thought = navigator(current_thought, memory)
        self.assertEqual(next_thought.shape, (1, self.hidden_size))

    def test_04_training_step(self):
        """Tests one forward and backward pass on the full model."""
        self.model.train()
        optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=1e-5)
        
        batch_items = [self.dataset[i] for i in range(2)]
        prompts = [f"Question: {item['question']}\n\nSolution:" for item in batch_items]
        full_texts = [f"{prompts[j]} {item['answer']}" for j, item in enumerate(batch_items)]
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        full_inputs = self.tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        teacher_thoughts_batch_embeds = [generate_structured_teacher_thoughts(item['question'], item['answer'], self.tokenizer, self.device, self.dtype, self.model.base_model.get_input_embeddings())[0] for item in batch_items]
        
        outputs = self.model(input_ids=full_inputs.input_ids, attention_mask=full_inputs.attention_mask, prompt_lengths=prompt_lengths, teacher_thoughts_embeds=teacher_thoughts_batch_embeds, epsilon=0.5)
        loss = outputs['loss']
        
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertIsNotNone(loss.item())

    def test_05_evaluation_step(self):
        """Tests the autoregressive generation method."""
        prompts = ["Question: What is 2+2?\n\nSolution:", "Question: What is 3*5?\n\nSolution:"]
        generated_answers = self.model.generate_with_reasoning_batched(self.tokenizer, prompts, max_new_tokens=10)
        self.assertIsInstance(generated_answers, list)
        self.assertEqual(len(generated_answers), 2)
        self.assertIsInstance(generated_answers[0], str)

    def test_06_graphing(self):
        """Tests the plotting function."""
        try:
            plot_training_curves([1.0, 0.9], [0.5, 0.6], [0.8, 0.7], [0.2, 0.2], is_smoke_test=True)
        except Exception as e:
            self.fail(f"plot_training_curves failed with an exception: {e}")

def run_unit_tests():
    """Runs the comprehensive unit test suite."""
    print("\n💨 Running Comprehensive Unit Tests...")
    suite = unittest.TestSuite()
    test_names = sorted([name for name in dir(TestCognitiveArchitecture) if name.startswith('test_')])
    for name in test_names:
        suite.addTest(TestCognitiveArchitecture(name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All unit tests passed successfully!")
        print("="*60)
        return True
    else:
        print("\n❌ Unit tests failed. Aborting training.")
        print("="*60)
        return False

# ============================================================
# TRAINING AND EVALUATION
# ============================================================
def train():
    # --- Configuration ---
    num_epochs = 100; initial_batch_size = 16; target_effective_batch = 32; max_length = 512
    base_lr = 1e-5; navigator_lr = 2e-5; device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epsilon_start = 1.0; epsilon_end = 0.1; epsilon_decay_duration = 0.75
    thought_loss_weight = 0.2
    
    print("\n" + "="*60 + "\nCOCONUT Training v39 - Comprehensive Unit Tests\n" + "="*60)
    
    # --- Model Loading with Unsloth ---
    print("\n📚 Loading tokenizer and model with Unsloth...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='meta-llama/Meta-Llama-3-8B-Instruct',
        max_seq_length=max_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    print("🔧 Applying LoRA configuration to the Unsloth model...")
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
    )
    base_model.print_trainable_parameters()
    
    model = CognitiveGraphModel(base_model=base_model, dropout_rate=0.1)
    
    model.to(device)
    
    # --- Run Unit Tests ---
    if not run_unit_tests():
        return

    # --- Optimizer & Scheduler ---
    navigator_params = list(model.navigator.parameters())
    base_params = [p for p in model.base_model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit([{'params': navigator_params, 'lr': navigator_lr}, {'params': base_params, 'lr': base_lr, 'weight_decay': 0.01}], betas=(0.9, 0.95))
    
    print("\n📊 Loading GSM8K dataset for full training...")
    dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
    val_dataset = load_dataset("gsm8k", "main", split="test")
    
    num_training_steps = (min(len(dataset), 2000) // initial_batch_size) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
    
    # --- Training Loop ---
    print("\n🚀 Starting full training run...")
    batch_size = initial_batch_size
    best_val_accuracy = 0.0
    train_losses, lm_losses, nav_losses, val_accuracies = [], [], [], []
    epsilon_decay_epochs = int(num_epochs * epsilon_decay_duration)

    for epoch in range(num_epochs):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (epoch / epsilon_decay_epochs))
        print(f"\n📅 Epoch {epoch+1}/{num_epochs} (Scheduled Sampling ϵ = {epsilon:.3f})")
        
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
            teacher_thoughts_batch_embeds = [generate_structured_teacher_thoughts(item['question'], item['answer'], tokenizer, device, model.base_model.dtype, model.base_model.get_input_embeddings())[0] for item in batch_items]
            try:
                outputs = model(
                    input_ids=full_inputs.input_ids, 
                    attention_mask=full_inputs.attention_mask, 
                    prompt_lengths=prompt_lengths, 
                    teacher_thoughts_embeds=teacher_thoughts_batch_embeds,
                    thought_loss_weight=thought_loss_weight,
                    epsilon=epsilon
                )
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

train()
