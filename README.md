# COCONUT PPO Implementation for Llama 3-8B

This repository contains an implementation of COCONUT (Chain of Continuous Thought) with PPO-based adaptive reasoning, optimized for Llama 3-8B on A100 40GB GPUs.

## 📋 Overview

COCONUT enhances Large Language Models by introducing continuous latent reasoning spaces, allowing models to perform complex reasoning without being constrained to discrete language tokens. This implementation adds PPO-based reinforcement learning for adaptive reasoning depth.

### Key Features
- ✅ Continuous reasoning in latent space
- ✅ PPO-based adaptive navigation
- ✅ Memory-efficient training for A100 40GB
- ✅ Support for GSM8K dataset (as in original paper)
- ✅ Partially frozen model training for memory efficiency

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU with 40GB+ memory
- Hugging Face account with Llama 3 access

### Installation

```bash
# Clone repository
git clone https://github.com/arccoxx/coconut.git
cd coconut

# Install dependencies
pip install torch transformers datasets accelerate tqdm

# Login to Hugging Face
huggingface-cli login
