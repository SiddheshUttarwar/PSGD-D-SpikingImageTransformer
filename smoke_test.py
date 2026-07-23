import sys
sys.path.insert(0, 'd-sit')
import torch
import torch.nn as nn
from model import DSIT

# Build the model with CIFAR-100 config
model = DSIT(
    num_classes=100,
    embed_dim=256,
    depth=8,
    num_heads=8,
    T=4,
    img_size=32
)

# Quick forward pass test
x = torch.randn(2, 3, 32, 32)
logits = model(x)
print(f"Forward pass OK — Output shape: {logits.shape}")

# Quick backward pass test
labels = torch.tensor([5, 10])
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
print(f"Backward pass OK — Loss: {loss.item():.4f}")

# Verify dopamine tracker
model.d_tracker.update(logits.detach())
print(f"Dopamine D(t): {model.d_tracker.get_D():.6f}")
