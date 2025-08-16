"""Quick integration test for COCONUT PPO with Llama 3 - Fixed version"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_ppo import CoconutPPO
import traceback

def test_coconut_integration():
    try:
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens
        special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"   Added {num_added} special tokens")
        
        print("\n2. Loading Llama 3-8B model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_cache=False
        )
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"   Model loaded, embeddings resized to {len(tokenizer)}")
        
        print("\n3. Creating COCONUT PPO model...")
        model = CoconutPPO(
            base_model=base_model,
            latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
            start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
            end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
            eos_token_id=tokenizer.eos_token_id,
            hidden_size=4096,  # Llama 3-8B hidden size
            reasoning_dim=256,
            max_latent_steps=4
        )
        
        # FIX: Move navigator to CUDA
        model.navigator = model.navigator.cuda()
        print(f"   COCONUT model created and moved to CUDA")
        
        print("\n4. Testing forward pass...")
        test_text = "What is 2 + 2?"
        inputs = tokenizer(test_text, return_tensors='pt', max_length=64, truncation=True)
        
        # Move to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_trajectory=True
                )
        
        print(f"   Forward pass successful!")
        print(f"   Output shape: {outputs['logits'].shape}")
        print(f"   Trajectory length: {len(outputs['trajectory']['states'])}")
        
        print("\n5. Memory check...")
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   Allocated: {mem_allocated:.1f}GB")
        print(f"   Reserved: {mem_reserved:.1f}GB")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_coconut_integration()
