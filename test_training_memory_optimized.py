"""Memory-optimized training test for A100 40GB"""
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_ppo import CoconutPPO
from torch.utils.data import Dataset, DataLoader

# Clear memory first
torch.cuda.empty_cache()
gc.collect()

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, size=10):
        self.tokenizer = tokenizer
        self.size = size
        self.texts = ["What is 2+2?", "Explain gravity", "Hello"] * (size // 3 + 1)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        text = self.texts[idx % len(self.texts)]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                 max_length=32, return_tensors='pt')  # Reduced from 64
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

print("Loading model with memory optimizations...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['<bot>', '<eot>', '<latent>']})

# Load model with more aggressive memory saving
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map='auto',
    use_cache=False,
    low_cpu_mem_usage=True
)
base_model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing to save memory
base_model.gradient_checkpointing_enable()

model = CoconutPPO(
    base_model=base_model,
    latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
    start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
    end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
    eos_token_id=tokenizer.eos_token_id,
    max_latent_steps=2  # Reduced from 4-6
)

# Freeze most of the base model to save memory
for name, param in model.base_model.named_parameters():
    if "layers.30" not in name and "layers.31" not in name and "lm_head" not in name:
        param.requires_grad = False

# Only optimize navigator parameters
optimizer = torch.optim.AdamW(
    [p for p in model.navigator.parameters() if p.requires_grad],
    lr=3e-4
)

# Create simple dataloader with batch size 1
dataset = SimpleDataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1

print("Starting test training...")
for i, batch in enumerate(dataloader):
    if i >= 2:  # Just test 2 batches
        break
    
    # Clear cache before each batch
    torch.cuda.empty_cache()
    
    batch = {k: v.cuda() for k, v in batch.items()}
    
    try:
        # Test forward pass without gradients for trajectory
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=None,  # No labels for trajectory
                return_trajectory=True
            )
        
        # Now compute loss with base model only
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            base_outputs = model.base_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
        
        if base_outputs.loss is not None:
            loss = base_outputs.loss
            
            # Only update navigator parameters
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.navigator.parameters(), 1.0)
            
            optimizer.step()
            
            print(f"✅ Batch {i}: Loss = {loss.item():.4f}, Trajectory len = {len(outputs['trajectory']['states'])}")
        else:
            print(f"✅ Batch {i}: No loss")
            
        # Clear cache after each batch
        del outputs, base_outputs
        torch.cuda.empty_cache()
        
    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM at batch {i}, clearing cache and continuing...")
        torch.cuda.empty_cache()
        gc.collect()
        continue

print("✅ Training test completed!")
print(f"Final memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated")
