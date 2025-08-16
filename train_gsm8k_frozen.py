"""
COCONUT PPO Training on GSM8K Dataset (from paper)
Partially frozen model for A100 40GB memory constraints
Only trains: navigator + last 2 transformer layers + lm_head
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from coconut_ppo import CoconutPPO
import logging
from tqdm import tqdm
import numpy as np
import json
import gc
import os
from datetime import datetime

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSM8KDataset(Dataset):
    """GSM8K dataset as used in the COCONUT paper"""
    def __init__(self, tokenizer, split='train', max_samples=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load GSM8K dataset
        logger.info(f"Loading GSM8K {split} split...")
        dataset = load_dataset("gsm8k", "main", split=split)
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = []
        for item in dataset:
            # Format as in COCONUT paper: question -> answer with chain of thought
            question = item['question']
            answer = item['answer']
            
            # Create input text
            input_text = f"Question: {question}\nAnswer: Let me solve this step by step.\n{answer}"
            self.data.append(input_text)
        
        logger.info(f"Loaded {len(self.data)} GSM8K examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class FrozenCoconutTrainer:
    """Trainer for COCONUT with partially frozen model"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Freeze strategy: Only train navigator + last 2 layers + lm_head
        self.freeze_model()
        
        # Create optimizer only for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=0.01
        )
        
        # Gradient accumulation
        self.grad_accum_steps = config['gradient_accumulation_steps']
        self.effective_batch_size = config['batch_size'] * self.grad_accum_steps
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def freeze_model(self):
        """Freeze most of the model, keeping only essential parts trainable"""
        trainable_count = 0
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Keep trainable: navigator, last 2 layers, lm_head
            if any(x in name for x in ["navigator", "layers.30", "layers.31", "lm_head"]):
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        total = trainable_count + frozen_count
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {total/1e9:.2f}B")
        logger.info(f"  Trainable: {trainable_count/1e6:.1f}M ({trainable_count/total*100:.1f}%)")
        logger.info(f"  Frozen: {frozen_count/1e9:.2f}B ({frozen_count/total*100:.1f}%)")
    
    def compute_rewards(self, loss, trajectory):
        """Compute rewards for PPO"""
        # Base reward from loss
        base_reward = -loss.item() if loss > 0 else 0
        
        # Length penalty to encourage efficiency
        length_penalty = -0.01 * len(trajectory['states'])
        
        # Total reward
        reward = base_reward + length_penalty
        
        return reward
    
    def train_step(self, batch):
        """Single training step"""
        try:
            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Get trajectory without gradients
                with torch.no_grad():
                    traj_outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=None,  # No labels for trajectory collection
                        return_trajectory=True
                    )
                
                # Compute task loss with gradients
                task_outputs = self.model.base_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            
            # Extract outputs
            loss = task_outputs.loss if hasattr(task_outputs, 'loss') else torch.tensor(0.0)
            trajectory = traj_outputs.get('trajectory', {'states': []})
            
            # Compute reward
            reward = self.compute_rewards(loss, trajectory)
            
            # Backward pass
            if loss > 0:
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()
                
                # Update weights after accumulation
                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Increment step
            self.global_step += 1
            
            # Clear cache periodically
            if self.global_step % 50 == 0:
                torch.cuda.empty_cache()
            
            return {
                'loss': loss.item() if loss > 0 else 0,
                'reward': reward,
                'traj_len': len(trajectory['states']),
                'memory_gb': torch.cuda.memory_allocated() / 1e9
            }
            
        except Exception as e:
            logger.error(f"Error in training step {self.global_step}: {e}")
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            return None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_rewards = []
        epoch_traj_lens = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            metrics = self.train_step(batch)
            
            if metrics:
                epoch_losses.append(metrics['loss'])
                epoch_rewards.append(metrics['reward'])
                epoch_traj_lens.append(metrics['traj_len'])
                
                # Update progress bar
                if len(epoch_losses) > 0:
                    avg_loss = np.mean([l for l in epoch_losses[-100:] if l > 0]) if any(epoch_losses[-100:]) else 0
                    avg_reward = np.mean(epoch_rewards[-100:]) if epoch_rewards else 0
                    avg_traj_len = np.mean(epoch_traj_lens[-100:]) if epoch_traj_lens else 0
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'reward': f'{avg_reward:.3f}',
                        'traj': f'{avg_traj_len:.1f}',
                        'mem': f'{metrics["memory_gb"]:.1f}GB'
                    })
                
                # Save checkpoint
                if self.global_step % self.config['save_steps'] == 0:
                    self.save_checkpoint(epoch, avg_loss)
                
                # Log metrics
                if self.global_step % self.config['log_steps'] == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={avg_loss:.4f}, "
                        f"reward={avg_reward:.3f}, "
                        f"traj_len={avg_traj_len:.1f}, "
                        f"mem={metrics['memory_gb']:.1f}GB"
                    )
        
        # Epoch summary
        epoch_avg_loss = np.mean([l for l in epoch_losses if l > 0]) if epoch_losses else 0
        epoch_avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        epoch_avg_traj = np.mean(epoch_traj_lens) if epoch_traj_lens else 0
        
        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
        logger.info(f"  Average reward: {epoch_avg_reward:.3f}")
        logger.info(f"  Average trajectory length: {epoch_avg_traj:.1f}")
        
        return epoch_avg_loss
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = f"checkpoints/coconut_gsm8k_step_{self.global_step}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(checkpoint, "checkpoints/best_model.pt")
            logger.info(f"Saved best model with loss {loss:.4f}")

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'batch_size': 1,
        'gradient_accumulation_steps': 8,  # Effective batch size = 8
        'learning_rate': 3e-4,
        'num_epochs': 3,
        'max_seq_length': 512,
        'max_latent_steps': 3,
        'save_steps': 100,
        'log_steps': 10,
        'max_train_samples': 1000,  # Start with subset for POC
    }
    
    logger.info("="*60)
    logger.info("COCONUT PPO Training on GSM8K (Proof of Concept)")
    logger.info("="*60)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for COCONUT
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens")
    
    # Load base model
    logger.info("\nLoading Llama 3-8B model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False,
        low_cpu_mem_usage=True
    )
    
    # Resize token embeddings and enable gradient checkpointing
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.gradient_checkpointing_enable()
    
    # Create COCONUT model
    logger.info("\nInitializing COCONUT PPO model...")
    model = CoconutPPO(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=256,
        max_latent_steps=config['max_latent_steps']
    )
    
    logger.info(f"Model initialized. Memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    
    # Create datasets
    logger.info("\nPreparing GSM8K dataset...")
    train_dataset = GSM8KDataset(
        tokenizer,
        split='train',
        max_samples=config['max_train_samples'],
        max_length=config['max_seq_length']
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Create trainer
    trainer = FrozenCoconutTrainer(model, config)
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train epoch
        avg_loss = trainer.train_epoch(train_loader, epoch + 1)
        
        # Save epoch checkpoint
        trainer.save_checkpoint(epoch + 1, avg_loss)
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best loss: {trainer.best_loss:.4f}")
    logger.info(f"Final memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    logger.info("="*60)

if __name__ == "__main__":
    # Install required package if not present
    try:
        import datasets
    except ImportError:
        print("Installing datasets package...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "-q"])
    
    main()
