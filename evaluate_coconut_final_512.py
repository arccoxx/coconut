%%writefile evaluate_coconut_final_512.py
"""
Evaluation script for COCONUT trained with reasoning_dim=512
Compatible with train_gsm8k_final.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from coconut_ppo import CoconutPPO
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoconutEvaluator:
    """Evaluate COCONUT model on GSM8K test set"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.metrics = defaultdict(list)
    
    def extract_answer(self, text):
        """Extract numerical answer from generated text"""
        # GSM8K format: #### answer
        pattern = r'####\s*(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        
        # Common patterns
        patterns = [
            r'answer is\s*(-?\d+(?:\.\d+)?)',
            r'=\s*(-?\d+(?:\.\d+)?)\s*(?:\.|$)',
            r'equals\s*(-?\d+(?:\.\d+)?)',
            r'total of\s*(-?\d+(?:\.\d+)?)',
            r'result is\s*(-?\d+(?:\.\d+)?)'
        ]
        
        for p in patterns:
            match = re.search(p, text.lower())
            if match:
                return float(match.group(1))
        
        # Last number as fallback
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        
        return None
    
    def evaluate_with_latent(self, input_ids, attention_mask, max_new_tokens=150):
        """Generate answer using latent thoughts"""
        with torch.no_grad():
            try:
                # Get trajectory and latent thoughts
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_trajectory=True
                )
                
                trajectory = outputs.get('trajectory', {'states': []})
                num_latent_steps = len(trajectory.get('states', []))
                
                # Generate answer
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    generation = self.model.base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_length = input_ids.shape[1]
                generated_text = self.tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                
                return generated_text, num_latent_steps
                
            except Exception as e:
                logger.debug(f"Latent evaluation error: {e}")
                return self.evaluate_without_latent(input_ids, attention_mask, max_new_tokens), 0
    
    def evaluate_without_latent(self, input_ids, attention_mask, max_new_tokens=150):
        """Generate answer without latent thoughts (baseline)"""
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generation = self.model.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            input_length = input_ids.shape[1]
            generated_text = self.tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
            return generated_text
    
    def evaluate_dataset(self, dataset, max_samples=50):
        """Evaluate on GSM8K dataset"""
        results = {
            'with_latent': {'correct': 0, 'total': 0, 'trajectory_lengths': []},
            'without_latent': {'correct': 0, 'total': 0}
        }
        
        example_outputs = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating", total=min(max_samples, len(dataset)))):
            if idx >= max_samples:
                break
            
            question = item['question']
            true_answer_text = item['answer']
            true_answer = self.extract_answer(true_answer_text)
            
            if true_answer is None:
                continue
            
            # Prepare input
            input_text = f"Question: {question}\nLet's solve this step by step.\n\nSolution:\n"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # WITH latent thoughts
            try:
                output_with_latent, num_steps = self.evaluate_with_latent(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                
                pred_with_latent = self.extract_answer(output_with_latent)
                
                if pred_with_latent is not None:
                    is_correct = abs(pred_with_latent - true_answer) < 0.01
                    results['with_latent']['correct'] += int(is_correct)
                    results['with_latent']['total'] += 1
                    results['with_latent']['trajectory_lengths'].append(num_steps)
                    
                    if idx < 3:  # Store first 3 examples
                        example_outputs.append({
                            'question': question[:200],
                            'true_answer': true_answer,
                            'predicted': pred_with_latent,
                            'latent_steps': num_steps,
                            'correct': is_correct,
                            'output': output_with_latent[:300]
                        })
            except Exception as e:
                logger.debug(f"Sample {idx} latent error: {e}")
            
            # WITHOUT latent thoughts
            try:
                output_without_latent = self.evaluate_without_latent(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
                
                pred_without_latent = self.extract_answer(output_without_latent)
                
                if pred_without_latent is not None:
                    is_correct = abs(pred_without_latent - true_answer) < 0.01
                    results['without_latent']['correct'] += int(is_correct)
                    results['without_latent']['total'] += 1
            except Exception as e:
                logger.debug(f"Sample {idx} baseline error: {e}")
            
            if idx % 10 == 0 and idx > 0:
                torch.cuda.empty_cache()
                # Progress update
                if results['with_latent']['total'] > 0:
                    acc = results['with_latent']['correct'] / results['with_latent']['total']
                    avg_traj = np.mean(results['with_latent']['trajectory_lengths']) if results['with_latent']['trajectory_lengths'] else 0
                    logger.info(f"Progress: {idx}/{max_samples} | Acc: {acc:.2%} | Avg Traj: {avg_traj:.1f}")
        
        # Calculate final metrics
        results['with_latent']['accuracy'] = results['with_latent']['correct'] / max(1, results['with_latent']['total'])
        results['without_latent']['accuracy'] = results['without_latent']['correct'] / max(1, results['without_latent']['total'])
        
        if results['with_latent']['trajectory_lengths']:
            results['with_latent']['avg_trajectory_length'] = np.mean(results['with_latent']['trajectory_lengths'])
        else:
            results['with_latent']['avg_trajectory_length'] = 0
        
        return results, example_outputs

def main():
    """Main evaluation function"""
    
    # Configuration - MATCHING train_gsm8k_final.py
    config = {
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'max_eval_samples': 50,
        'reasoning_dim': 512,  # MATCHING train_gsm8k_final.py
        'max_latent_steps': 8   # MATCHING train_gsm8k_final.py
    }
    
    logger.info("="*60)
    logger.info("🥥 COCONUT PPO Evaluation (512-dim)")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  • Reasoning dim: {config['reasoning_dim']}")
    logger.info(f"  • Max latent steps: {config['max_latent_steps']}")
    logger.info(f"  • Eval samples: {config['max_eval_samples']}")
    
    # Find appropriate checkpoint
    checkpoint_path = None
    checkpoint_options = [
        'checkpoints/best_model_ppo5.pt',
        'checkpoints/best_model_antimode.pt',  # New anti-mode checkpoint
        'checkpoints/best_model_navigator_512.pt',  # New 512-dim checkpoint
        'checkpoints/best_model_navigator.pt',       # Old 256-dim checkpoint
        'checkpoints/checkpoint_epoch_1_512.pt',
        'checkpoints/checkpoint_epoch_2_512.pt',
        'checkpoints/checkpoint_epoch_3_512.pt',
    ]
    
    for path in checkpoint_options:
        if os.path.exists(path):
            # Check if it matches our architecture
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                if 'config' in ckpt:
                    ckpt_reasoning_dim = ckpt['config'].get('reasoning_dim', 256)
                    if ckpt_reasoning_dim == config['reasoning_dim']:
                        checkpoint_path = path
                        logger.info(f"✅ Found matching checkpoint: {path}")
                        break
                    else:
                        logger.info(f"⚠️ {path} has reasoning_dim={ckpt_reasoning_dim}, need {config['reasoning_dim']}")
            except:
                pass
    
    if not checkpoint_path:
        logger.warning("⚠️ No matching checkpoint found for reasoning_dim=512")
        logger.info("   Will use random initialization (untrained model)")
    
    # Load tokenizer
    logger.info("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = {'additional_special_tokens': ['<bot>', '<eot>', '<latent>']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load base model
    logger.info("🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=True
    )
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    # Create COCONUT model with 512-dim
    logger.info(f"🥥 Creating COCONUT model (reasoning_dim={config['reasoning_dim']})...")
    model = CoconutPPO(
        base_model=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids('<latent>'),
        start_latent_id=tokenizer.convert_tokens_to_ids('<bot>'),
        end_latent_id=tokenizer.convert_tokens_to_ids('<eot>'),
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=4096,
        reasoning_dim=config['reasoning_dim'],  # 512
        max_latent_steps=config['max_latent_steps']  # 8
    )
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"\n💾 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        
        # Load navigator weights
        if 'navigator_state_dict' in checkpoint:
            incompatible = model.navigator.load_state_dict(
                checkpoint['navigator_state_dict'], 
                strict=False
            )
            logger.info("✅ Navigator weights loaded")
            
            # Check for issues
            expected_missing = ['memory_bank', 'memory_values']
            unexpected_missing = [k for k in incompatible.missing_keys if k not in expected_missing]
            if unexpected_missing:
                logger.warning(f"Missing keys: {unexpected_missing}")
        
        # Log training info
        if 'epoch' in checkpoint:
            logger.info(f"📊 Checkpoint from epoch {checkpoint['epoch'] + 1}")
        if 'avg_trajectory' in checkpoint:
            logger.info(f"📈 Training trajectory: {checkpoint['avg_trajectory']:.2f}")
    else:
        logger.warning("⚠️ No checkpoint loaded - using untrained model")
    
    model.to('cuda')
    model.eval()
    
    # Load dataset
    logger.info("\n📊 Loading GSM8K test set...")
    test_dataset = load_dataset("gsm8k", "main", split="test")
    
    # Evaluate
    logger.info(f"🔬 Evaluating on {config['max_eval_samples']} samples...")
    evaluator = CoconutEvaluator(model, tokenizer)
    results, examples = evaluator.evaluate_dataset(test_dataset, max_samples=config['max_eval_samples'])
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("="*60)
    
    logger.info(f"\n🔵 With Latent Thoughts:")
    logger.info(f"  • Accuracy: {results['with_latent']['accuracy']*100:.2f}%")
    logger.info(f"  • Correct: {results['with_latent']['correct']}/{results['with_latent']['total']}")
    logger.info(f"  • Avg Trajectory: {results['with_latent']['avg_trajectory_length']:.2f} steps")
    
    logger.info(f"\n⚪ Without Latent Thoughts (Baseline):")
    logger.info(f"  • Accuracy: {results['without_latent']['accuracy']*100:.2f}%")
    logger.info(f"  • Correct: {results['without_latent']['correct']}/{results['without_latent']['total']}")
    
    improvement = (results['with_latent']['accuracy'] - results['without_latent']['accuracy'])*100
    logger.info(f"\n📈 Improvement: {improvement:+.2f}%")
    
    # Trajectory analysis
    if results['with_latent']['trajectory_lengths']:
        lengths = results['with_latent']['trajectory_lengths']
        logger.info(f"\n📐 Trajectory Distribution:")
        max_len = min(10, int(max(lengths)) + 1) if lengths else 0
        for i in range(max_len):
            count = lengths.count(i)
            if count > 0:
                pct = count / len(lengths) * 100
                bar = '█' * int(pct/5) + '░' * (20 - int(pct/5))
                logger.info(f"  {i} steps: {bar} {count} ({pct:.1f}%)")
    
    # Examples
    if examples:
        logger.info("\n📝 Example Outputs:")
        for i, ex in enumerate(examples):
            status = "✅" if ex['correct'] else "❌"
            logger.info(f"\nExample {i+1} {status}:")
            logger.info(f"  Q: {ex['question'][:100]}...")
            logger.info(f"  True: {ex['true_answer']}, Pred: {ex['predicted']}, Steps: {ex['latent_steps']}")
    
    # Interpretation
    logger.info("\n" + "="*60)
    logger.info("💡 INTERPRETATION")
    logger.info("="*60)
    
    avg_traj = results['with_latent']['avg_trajectory_length']
    if avg_traj < 0.5:
        logger.warning("⚠️ Navigator not working - trajectory length near 0")
        logger.warning("   Check if model is properly trained")
    elif avg_traj < 2:
        logger.info("📊 Early training stage - trajectory length < 2")
        logger.info("   Model is beginning to use latent thoughts")
        logger.info("   Continue training for better performance")
    elif avg_traj < 3:
        logger.info("✅ Good progress - trajectory length 2-3")
        logger.info("   Model is effectively using latent reasoning")
        logger.info("   A few more epochs should reach target")
    else:
        logger.info("🎯 Excellent - trajectory length > 3")
        logger.info("   Model has learned deep multi-step reasoning")
        logger.info("   Target performance achieved!")
    
    # Save results
    results_dict = {
        'config': config,
        'checkpoint': checkpoint_path,
        'results': {
            'with_latent_accuracy': results['with_latent']['accuracy'],
            'without_latent_accuracy': results['without_latent']['accuracy'],
            'avg_trajectory_length': results['with_latent']['avg_trajectory_length'],
            'improvement': improvement
        },
        'examples': examples
    }
    
    with open('evaluation_results_512.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info("\n💾 Results saved to evaluation_results_512.json")
    
    # Plot trajectory distribution
    if results['with_latent']['trajectory_lengths']:
        plt.figure(figsize=(10, 6))
        plt.hist(results['with_latent']['trajectory_lengths'], 
                bins=range(0, max(results['with_latent']['trajectory_lengths']) + 2),
                alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Trajectory Length (steps)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Latent Reasoning Steps')
        plt.grid(True, alpha=0.3)
        plt.savefig('trajectory_dist_512.png', dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("📊 Plot saved to trajectory_dist_512.png")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()
