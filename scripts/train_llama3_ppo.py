#!/usr/bin/env python3
"""
Training script for COCONUT PPO with Llama 3-8B on A100
"""

import os
import sys
import torch
import logging
from pathlib import Path
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    set_seed
)
import yaml
import wandb
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from coconut_ppo import CoconutPPO
from ppo_trainer import PPOTrainer
from debug_utils import CoconutDebugger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
    
    if gpu_memory < 40:
        logger.warning(f"GPU has less than 40GB memory ({gpu_memory:.1f}GB)")
    
    return gpu_name, gpu_memory

def load_model_and_tokenizer(config):
    """Load Llama 3-8B with memory optimizations"""
    logger.info("Loading Llama 3-8B model...")
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        config['model']['name'],
        use_fast=True
    )
    
    # Add special tokens for COCONUT
    special_tokens = {
        'additional_special_tokens': [
            '<bot>', '<eot>', '<latent>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    model = LlamaForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.bfloat16,  # Use bfloat16 for A100
        device_map='auto',  # Automatic device placement
        use_cache=False,  # Disable KV cache for training
    )
    
    # Resize embeddings for special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Get token IDs
    latent_token_id = tokenizer.convert_tokens_to_ids('<latent>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<bot>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<eot>')
    eos_token_id = tokenizer.eos_token_id
    
    # Create COCONUT PPO model
    coconut_model = CoconutPPO(
        base_model=model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id,
        hidden_size=config['model']['hidden_size'],
        reasoning_dim=config['model']['reasoning_dim'],
        max_latent_steps=config['model']['max_latent_steps']
    )
    
    logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in coconut_model.parameters()) / 1e9:.2f}B")
    
    return coconut_model, tokenizer

def create_dataloaders(tokenizer, config):
    """Create data loaders (simplified for demo)"""
    # TODO: Replace with actual dataset loading
    from torch.utils.data import Dataset
    
    class DemoDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Demo data - replace with real data
            text = f"Question: What is 2 + 2? Answer:"
            inputs = tokenizer(
                text,
                max_length=config['data']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['input_ids'].squeeze()
            }
    
    train_dataset = DemoDataset(1000)
    val_dataset = DemoDataset(100)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader

def main():
    # Load configuration
    with open('configs/llama3_ppo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(42)
    
    # Check GPU
    gpu_name, gpu_memory = check_gpu()
    
    # Initialize wandb
    if config['debug']['wandb_project']:
        wandb.init(
            project=config['debug']['wandb_project'],
            config=config
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Move model to GPU
    device = torch.device('cuda')
    model = model.to(device)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(tokenizer, config)
    
    # Initialize trainer and debugger
    trainer = PPOTrainer(model, config['ppo'], device)
    debugger = CoconutDebugger(log_to_wandb=config['debug']['wandb_project'] is not None)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Monitor memory
            if batch_idx % 10 == 0:
                allocated, reserved = debugger.monitor_memory_usage()
            
            # Training step
            metrics = trainer.train_step(batch)
            
            # Log metrics
            if global_step % config['training']['logging_steps'] == 0:
                logger.info(f"Step {global_step}: {metrics}")
                if wandb.run:
                    wandb.log(metrics, step=global_step)
            
            # Debug trajectory
            if config['debug']['log_trajectories'] and global_step % 100 == 0:
                # Get a sample trajectory for visualization
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    outputs = model(
                        input_ids=sample_batch['input_ids'].to(device),
                        return_trajectory=True
                    )
                    debugger.log_trajectory(outputs['trajectory'], global_step)
            
            # Check gradients
            if config['debug']['check_gradients'] and global_step % 50 == 0:
                debugger.check_gradient_flow(model)
            
            # Save checkpoint
            if global_step % config['training']['save_steps'] == 0:
                checkpoint_path = f"checkpoints/coconut_llama3_step_{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'step': global_step,
                    'config': config
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            global_step += 1
            
            # Early stopping for demo
            if global_step > 100:
                break
    
    logger.info("Training completed!")
    
    # Final save
    torch.save(model.state_dict(), "checkpoints/coconut_llama3_final.pt")
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()
