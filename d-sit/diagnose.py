"""
Quick diagnostic to run on Colab BEFORE full training.
Prints firing rates, gradient norms, and logit statistics to pinpoint
where the network is dying.

Usage: python d-sit/diagnose.py
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'd-sit')

from model import DSIT
from optimizer import ProximalAdamW

def diagnose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = DSIT(num_classes=100, embed_dim=256, depth=8, num_heads=8, T=4, img_size=32).to(device)

    # =========================================================
    # TEST 1: Forward pass WITHOUT AMP
    # =========================================================
    print("=" * 60)
    print("TEST 1: Forward pass (NO AMP)")
    print("=" * 60)
    x = torch.randn(4, 3, 32, 32, device=device)

    # Hook to capture intermediate activations
    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out.is_floating_point():
                activations[name] = {
                    'mean': out.mean().item(),
                    'std': out.std().item(),
                    'firing_rate': (out > 0).float().mean().item(),
                    'dtype': str(out.dtype),
                    'shape': list(out.shape),
                }
        return hook

    hooks = []
    for name, module in model.named_modules():
        if 'lif' in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        logits = model(x)

    print(f"\nLogits: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
    print(f"Logit range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Logit dtype: {logits.dtype}")
    argmax_counts = torch.bincount(logits.argmax(dim=1), minlength=100)
    print(f"Predictions: {logits.argmax(dim=1).tolist()}")
    print(f"All same class? {(logits.argmax(dim=1) == logits.argmax(dim=1)[0]).all().item()}")

    print(f"\n{'Layer':<50} {'Firing Rate':>12} {'Mean':>10} {'Std':>10} {'Dtype':>10}")
    print("-" * 92)
    for name, stats in sorted(activations.items()):
        print(f"{name:<50} {stats['firing_rate']:>12.4f} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['dtype']:>10}")

    for h in hooks:
        h.remove()

    # =========================================================
    # TEST 2: Backward pass gradient flow (NO AMP)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("TEST 2: Backward pass gradient flow (NO AMP)")
    print("=" * 60)

    model.zero_grad()
    x = torch.randn(4, 3, 32, 32, device=device)
    labels = torch.randint(0, 100, (4,), device=device)

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    print(f"\nLoss: {loss.item():.4f}")
    print(f"\n{'Parameter':<50} {'Grad Norm':>12} {'Weight Norm':>12} {'Grad/Weight':>12}")
    print("-" * 86)
    total_zero_grad = 0
    total_params = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            wn = p.data.norm().item()
            ratio = gn / (wn + 1e-8)
            total_params += 1
            if gn < 1e-10:
                total_zero_grad += 1
            # Only print important layers
            if any(k in name for k in ['conv', 'fc', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'head.', 'v_th', 'attn_scale']):
                print(f"{name:<50} {gn:>12.6f} {wn:>12.4f} {ratio:>12.8f}")

    print(f"\nZero-gradient params: {total_zero_grad}/{total_params}")

    # =========================================================
    # TEST 3: Forward pass WITH AMP (to check if AMP breaks it)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("TEST 3: Forward pass WITH AMP")
    print("=" * 60)

    model.zero_grad()
    model.reset_all_states()
    x = torch.randn(4, 3, 32, 32, device=device)
    labels = torch.randint(0, 100, (4,), device=device)

    with torch.cuda.amp.autocast(enabled=True):
        logits_amp = model(x)
        loss_amp = nn.CrossEntropyLoss()(logits_amp, labels)

    scaler = torch.cuda.amp.GradScaler()
    scaler.scale(loss_amp).backward()

    print(f"Logits (AMP): mean={logits_amp.mean().item():.4f}, std={logits_amp.std().item():.4f}")
    print(f"Logit range: [{logits_amp.min().item():.3f}, {logits_amp.max().item():.3f}]")
    print(f"Logit dtype: {logits_amp.dtype}")
    print(f"Loss (AMP): {loss_amp.item():.4f}")
    print(f"Loss is NaN: {torch.isnan(loss_amp).item()}")
    print(f"Loss is Inf: {torch.isinf(loss_amp).item()}")

    # Check for NaN/zero gradients under AMP
    nan_grads = 0
    zero_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                nan_grads += 1
                print(f"  NaN gradient: {name}")
            if p.grad.norm().item() < 1e-10:
                zero_grads += 1

    print(f"\nAMP NaN gradients: {nan_grads}")
    print(f"AMP Zero gradients: {zero_grads}/{total_params}")

    # =========================================================
    # TEST 4: 5 training steps to check if loss decreases
    # =========================================================
    print(f"\n{'=' * 60}")
    print("TEST 4: 5 training steps (NO AMP)")
    print("=" * 60)

    model2 = DSIT(num_classes=100, embed_dim=256, depth=8, num_heads=8, T=4, img_size=32).to(device)
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(5):
        model2.zero_grad()
        x = torch.randn(8, 3, 32, 32, device=device)
        labels = torch.randint(0, 100, (8,), device=device)

        logits = model2(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        model2.d_tracker.update(logits.detach())

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        print(f"  Step {step+1}: loss={loss.item():.4f}, acc={acc*100:.1f}%, "
              f"logit_std={logits.std().item():.4f}, D(t)={model2.d_tracker.get_D():.6f}")

    print("\n✅ Diagnostics complete.")
    print("\nIf TEST 3 shows NaN/zero gradients but TEST 2 doesn't → AMP is the problem.")
    print("If TEST 2 shows zero gradients → Surrogate gradient is broken.")
    print("If TEST 4 loss doesn't decrease → Optimization is broken.")
    print("If firing rates are 0.0 in any layer → Dead neuron problem.")

if __name__ == '__main__':
    diagnose()
