"""Prepare training data for COCONUT PPO"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: "Question: {question}\nAnswer:"
        if isinstance(item, dict):
            if 'question' in item and 'answer' in item:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            else:
                text = str(item)
        else:
            text = str(item)
        
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

def load_gsm8k():
    """Load GSM8K dataset"""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    
    # Format for our needs
    train_data = []
    for item in dataset['train']:
        train_data.append({
            'question': item['question'],
            'answer': item['answer']
        })
    
    test_data = []
    for item in dataset['test']:
        test_data.append({
            'question': item['question'],
            'answer': item['answer']
        })
    
    print(f"Loaded {len(train_data)} training examples, {len(test_data)} test examples")
    return train_data, test_data

def create_dataloaders(tokenizer, batch_size=2):
    """Create training and validation dataloaders"""
    
    # Load data
    train_data, test_data = load_gsm8k()
    
    # Create datasets
    train_dataset = ReasoningDataset(train_data[:1000], tokenizer)  # Start with 1000 samples
    val_dataset = ReasoningDataset(test_data[:100], tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Test it
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = create_dataloaders(tokenizer, batch_size=2)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
