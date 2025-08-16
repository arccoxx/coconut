"""
Fully memory-optimized training for COCONUT on A100 40GB
Uses gradient accumulation, frozen layers, and aggressive memory management
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_ppo import CoconutPPO
from prepare_dataset import create_dataloaders
import logging
from tqdm import tqdm
import numpy as np
import gc
import os

# Set memory management environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientPPOTrainer:
    """PPO trainer optimized for limited GPU memory"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Only optimize navigator and last few layers
        trainable_params = []
        for name, param in model.named_parameters():
            if "navigator" in name or "layers.31" in name or "lm_head" in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        logger.info(f"Training {len(trainable_params)} parameter groups")
        
        # Use memory-efficient optimizer
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config['lr_navigator'])
        
        # Gradient accumulation for effective larger batch
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 8)
        self.actual_batch_size = config['batch_size'] * self.gradient_accumulation_steps
        
        # PPO parameters
        self.clip_epsilon = config['clip_epsilon']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        
        # Simple rollout buffer
        self.rollout_buffer = []
        self.max_buffer_size = 10  # Keep small
    
    def train_step(self, batch, step_idx):
        """Single training step with memory management"""
        try:
            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward pass for trajectory (no gradients)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    traj_outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=None,
                        return_trajectory=True
                    )
            
            # Get trajectory
            trajectory = traj_outputs.get('trajectory', {'states': [], 'values': []})
            
            # Forward pass for loss (with gradients only for trainable params)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_outputs = self.model.base_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            
            loss = loss_outputs.loss if hasattr(loss_outputs, 'loss') else None
            
            if loss is not None:
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Update weights every N steps
                if (step_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Store minimal trajectory info
            if len(trajectory['states']) > 0 and len(self.rollout_buffer) < self.max_buffer_size:
                self.rollout_buffer.append({
                    'length': len(trajectory['states']),
                    'loss': loss.item() if loss else 0
                })
            
            # Clear intermediate tensors
            del traj_outputs, loss_outputs
            torch.cuda.empty_cache()
            
            return {
                'loss': loss.item() if loss else 0,
                'traj_len': len(trajectory['states'])
            }
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at step {step_idx}, clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            self.optimizer.zero_grad()
            return {'loss': 0, 'traj_len': 0}

def main():
    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Configuration for memory-constrained training
    config = {
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'batch_size': 1,  # Minimum batch size
        'gradient_accumulation_steps': 8,  # Effective batch = 8
        'lr_navigator': 3e-4,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_seq_length': 256,  # Reduced sequence length
        'num_epochs': 2,  # Fewer epochs for testing
        'save_steps': 100,
        'logging_steps': 10
    }
    
    # Load model and tokenizer
    logger.info("Loading model with memory optimizations...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load with memory optimizations
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False,
        low_cpu_mem_usage=True
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()
    
    # Create COCONUT model with reduced latent steps
    model = CoconutPPO(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=128,  # Reduced from 256
        max_latent_steps=2   # Reduced from 4-6
    )
    
    logger.info(f"Model ready. Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    
    # Create simple dataset for testing
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.prompts = [
                "What is 2 + 2?",
                "Explain gravity in simple terms.",
                "Write a haiku about coding.",
                "What is the capital of France?",
                "How does photosynthesis work?"
            ]
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            text = self.prompts[idx % len(self.prompts)]
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config['max_seq_length'],
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze()
            }
    
    # Create data loader
    train_dataset = SimpleDataset(200)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=False
    )
    
    # Create trainer
    trainer = MemoryEfficientPPOTrainer(model, config)
    
    # Training loop
    logger.info("Starting memory-efficient training...")
    global_step = 0
    
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        epoch_traj_lens = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Train step
            metrics = trainer.train_step(batch, global_step)
            
            epoch_losses.append(metrics['loss'])
            epoch_traj_lens.append(metrics['traj_len'])
            
            # Update progress bar
            if len(epoch_losses) > 0:
                progress_bar.set_postfix({
                    'loss': np.mean([l for l in epoch_losses[-10:] if l > 0]) if any(epoch_losses[-10:]) else 0,
                    'traj_len': np.mean(epoch_traj_lens[-10:]) if epoch_traj_lens else 0,
                    'mem_GB': torch.cuda.memory_allocated() / 1e9
                })
            
            # Periodic memory cleanup
            if global_step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Save checkpoint
            if global_step % config['save_steps'] == 0 and global_step > 0:
                checkpoint = {
                    'step': global_step,
                    'navigator_state_dict': model.navigator.state_dict(),
                    'metrics': {'loss': np.mean(epoch_losses), 'traj_len': np.mean(epoch_traj_lens)}
                }
                torch.save(checkpoint, f'checkpoint_step_{global_step}.pt')
                logger.info(f"Saved checkpoint at step {global_step}")
            
            global_step += 1
            
            # Early stopping for testing
            if global_step > 100:
                break
        
        logger.info(f"Epoch {epoch} completed. Avg loss: {np.mean([l for l in epoch_losses if l > 0]):.4f}")
        
        if global_step > 100:
            break
    
    logger.info("Training completed!")
    logger.info(f"Final memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

if __name__ == "__main__":
    main()
