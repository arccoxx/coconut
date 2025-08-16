"""
Debugging utilities for COCONUT PPO
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import logging
import wandb

logger = logging.getLogger(__name__)

class CoconutDebugger:
    """Debug and monitor COCONUT training"""
    
    def __init__(self, log_to_wandb: bool = True):
        self.log_to_wandb = log_to_wandb
        self.trajectory_history = []
        self.loss_history = []
        self.reward_history = []
        
    def log_trajectory(self, trajectory: Dict, step: int):
        """Log trajectory information"""
        num_steps = len(trajectory['states'])
        
        metrics = {
            'trajectory/length': num_steps,
            'trajectory/avg_value': torch.stack(trajectory['values']).mean().item(),
            'trajectory/final_value': trajectory['values'][-1].item() if num_steps > 0 else 0
        }
        
        # Log action distribution
        if 'actions' in trajectory and len(trajectory['actions']) > 0:
            actions = torch.stack(trajectory['actions'])
            unique, counts = torch.unique(actions, return_counts=True)
            for u, c in zip(unique, counts):
                metrics[f'trajectory/action_{u.item()}_freq'] = c.item() / num_steps
        
        if self.log_to_wandb:
            wandb.log(metrics, step=step)
        
        logger.info(f"Step {step}: Trajectory length={num_steps}")
        return metrics
    
    def check_gradient_flow(self, model: nn.Module):
        """Check for vanishing/exploding gradients"""
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                if param_norm > 100:
                    logger.warning(f"Large gradient in {name}: {param_norm}")
                elif param_norm < 1e-6:
                    logger.warning(f"Small gradient in {name}: {param_norm}")
        
        total_norm = total_norm ** 0.5
        avg_norm = total_norm / max(param_count, 1)
        
        logger.info(f"Gradient norm: total={total_norm:.4f}, avg={avg_norm:.4f}")
        return total_norm, avg_norm
    
    def visualize_reasoning_trajectory(self, trajectory: List[torch.Tensor], save_path: str = None):
        """Visualize reasoning path in 2D"""
        if len(trajectory) == 0:
            return
        
        # Convert to numpy
        positions = torch.stack(trajectory).cpu().numpy()
        
        # Reduce to 2D using PCA
        from sklearn.decomposition import PCA
        if positions.shape[1] > 2:
            pca = PCA(n_components=2)
            positions_2d = pca.fit_transform(positions)
        else:
            positions_2d = positions
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(positions_2d[:, 0], positions_2d[:, 1], 'b-', alpha=0.5, linewidth=2)
        plt.scatter(positions_2d[0, 0], positions_2d[0, 1], c='green', s=100, label='Start')
        plt.scatter(positions_2d[-1, 0], positions_2d[-1, 1], c='red', s=100, label='End')
        plt.scatter(positions_2d[1:-1, 0], positions_2d[1:-1, 1], c='blue', s=30, alpha=0.6)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Reasoning Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def monitor_memory_usage(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            
            logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            if allocated > 35:  # Warning at 35GB for A100 40GB
                logger.warning("High GPU memory usage!")
            
            return allocated, reserved
        return 0, 0
