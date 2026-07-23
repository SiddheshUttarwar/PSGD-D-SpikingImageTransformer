# Tripartite Inhibitory Attention (Neuro-Glial Transformer)

This repository contains the implementation of the **Neuro-Glial Transformer**, an advanced Spiking Neural Network (SNN) architecture designed to break the 80% accuracy barrier on CIFAR-100 by solving the two fundamental bottlenecks of Spiking Transformers: Spatial Routing and Credit Assignment.

## Architecture Breakthroughs

### 1. Dopaminergic Lateral Inhibitory Attention (DLIA)
Traditional Transformers rely on a floating-point Softmax operation, which is fundamentally incompatible with true SNNs.
We replace the standard $O(N^2)$ static attention matrix with an active, recurrent grid of spiking "Attention Interneurons" (`attention.py`).
- **Spiking Softmax**: Interneurons use lateral inhibition to suppress neighboring signals. This naturally forces the attention matrix into a mathematically strict $L_0$ sparse state without relying on exponential functions.
- **Dopaminergic Gating**: The strength of lateral inhibition is gated by the localized Dopamine RPE, natively shifting the network between Exploitation (hyper-sparse focus) and Exploration (diffuse broad attention).

### 2. Dopamine-Gated Astrocytic Syncytium (DATA)
Biological brains do not rely on a single global scalar for learning in early visual layers. 
- **Calcium Diffusion**: We simulate an astrocytic syncytium (`astrocyte.py`) that integrates local spiking activity into a continuous Calcium ($Ca^{2+}$) state. This state physically diffuses across the 2D image grid using a spatial Laplacian PDE kernel.
- **Localized Credit Assignment**: The global Dopamine scalar is scaled by the local Astrocytic $Ca^{2+}$ concentration, instantly transforming the scalar into a **high-resolution localized spatial gradient matrix**. This allows deep surrogate gradients (`neurons.py`) to know exactly which spatial patch caused an error, solving the credit assignment bottleneck.

### 3. Hybrid ANN Dense Stem
To prevent catastrophic early information loss and maximize spatial geometry, the raw image pixels are processed via a continuous `HybridANNConvStem` (`model.py`), which outputs a dense $16 \times 16$ grid (256 tokens) before passing the signal to the SNN blocks.

## Training on Google Colab
The primary training notebook is provided: `DSIT_Train_CIFAR100.ipynb`

To train the model on Colab:
1. Open `DSIT_Train_CIFAR100.ipynb` in Google Colab.
2. Ensure you are connected to a high-RAM GPU instance (A100/V100 recommended).
3. Run all cells to initiate the 300-epoch training run.

## Repository Structure
- `d-sit/astrocyte.py` - Astrocytic diffusion and localized credit assignment.
- `d-sit/attention.py` - Spiking Softmax via DLIA.
- `d-sit/neurons.py` - Spiking neurons with Localized DA-PSG.
- `d-sit/model.py` - Hybrid ANN Dense Stem and Transformer Blocks.
- `d-sit/train.py` - PyTorch training loop and optimizers.
