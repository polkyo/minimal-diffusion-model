# minimal-diffusion-model

A minimal PyTorch implementation of diffusion models (denoising score matching)  
Generates synthetic data (e.g. MNIST), trains a small U‑Net or MLP to reverse diffusion.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib

## File structure
- `diffusion_model.py` – core classes and functions
- `train.py` – training loop
- `generate.py` – sample generation
- `utils.py` – data loading and visualization

## Usage
1. Train model:
   ```bash
   python train.py --dataset mnist --epochs 50 --batch-size 128
