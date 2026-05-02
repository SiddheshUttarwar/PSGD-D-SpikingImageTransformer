# PSGD-D: Dopaminergic Spiking Image Transformer

**A Mathematical Framework for Dopamine-Modulated Proximal Surrogate Gradients in Sparsity-Aware Spiking Transformers**

> Siddhesh Uttarwar

## Overview

This repository implements the **Dopaminergic Spiking Image Transformer (D-SIT)**, a biologically-inspired spiking neural network architecture that introduces:

- **Dopamine-Modulated Proximal Surrogate Gradient (DA-PSG):** An adaptive surrogate gradient whose width is dynamically controlled by a global Reward Prediction Error (RPE) signal, transitioning from broad exploratory gradients to precise Dirac-delta temporal locking as the network converges.
- **Heterogeneous Spiking Self-Attention (SSA):** Multi-head attention with binary exploration heads and ternary exploitation heads, computed without Softmax for neuromorphic hardware compatibility.
- **Proximal Sparsity Controller:** An ℓ₂,₁-norm group sparsity optimizer that dynamically prunes attention heads during training based on their temporal synchronization with the global reward.
- **Learnable Time-Delay Tokenizer:** Per-channel temporal phase shifts that enable temporal feature engineering without increasing the simulation window.

## Architecture

```
Input Image (B, 3, H, W)
    │
    ├── [T=4 timestep repetition]
    │
    ▼
Spiking Conv Stem (4-layer, BatchNorm + LIF)
    │
    ▼
Learnable Time-Delay Tokenizer
    │
    ▼
L × D-SIT Transformer Blocks
    ├── Heterogeneous SSA (8 binary + 4 ternary heads)
    │       └── No Softmax (spike dot-product)
    ├── Dynamic Head Pruning Gate
    └── Spiking MLP (4× expansion, Membrane Shortcuts)
    │
    ▼
Spatio-Temporal Average Pooling
    │
    ▼
Linear Classifier → Logits
```

## Quick Start

### Install Dependencies
```bash
pip install torch torchvision datasets tqdm
```

### Train on CIFAR-100
```bash
python d-sit/train.py --dataset cifar100 --batch_size 16 --accum_steps 4 --epochs 100
```

### Train on ImageNet-1K
```bash
python d-sit/train.py --dataset imagenet-1k --embed_dim 768 --depth 12 --num_heads 12 --img_size 224
```

### Resume from Checkpoint (Colab)
```bash
python d-sit/train.py --dataset cifar100 --resume dsit_latest.pth
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embed_dim` | 256 | Embedding dimension (256 for CIFAR/T4, 768 for ImageNet) |
| `--depth` | 8 | Number of transformer blocks |
| `--num_heads` | 8 | Number of attention heads |
| `--T` | 4 | Simulation timesteps |
| `--prox_lambda` | 1e-4 | Proximal sparsity strength |
| `--lr` | 1e-3 | Learning rate |

## Mathematical Foundation

The DA-PSG surrogate gradient adapts its width based on the dopaminergic signal:

$$\alpha(D) = \frac{\alpha_{base}}{1 + \kappa D(t)}$$

$$\sigma'_{DA}(u) = \frac{1}{2\alpha(D)} \left(1 + \frac{|u - V_{th}|}{\alpha(D)}\right)^{-2}$$

**Lemma 1 (Exploration):** When D(t) → 0, the surrogate expands to maximum width, guaranteeing gradient flow to all neurons and protecting heads from premature pruning.

**Lemma 2 (Exploitation):** When D(t) → ∞, the surrogate collapses to a Dirac delta, gating updates to exact spike times only, eliminating temporal aliasing.

## Project Structure

```
d-sit/
├── neurons.py      # DA-PSG autograd function, LIF neuron, DopamineTracker
├── tokenizer.py    # Learnable Time-Delay Tokenizer
├── attention.py    # Heterogeneous Spiking Self-Attention (binary + ternary)
├── model.py        # Full D-SIT: Conv Stem, S-MLP, Transformer Blocks
├── optimizer.py    # ProximalAdamW with ℓ₂,₁ head pruning
├── train.py        # Training loop (AMP, gradient accumulation, scheduling)
└── __init__.py
```

## Citation

```bibtex
@article{uttarwar2026dsit,
  title={A Mathematical Framework for Dopamine-Modulated Proximal Surrogate Gradients in Sparsity-Aware Spiking Transformers},
  author={Uttarwar, Siddhesh},
  year={2026}
}
```

## License

MIT
