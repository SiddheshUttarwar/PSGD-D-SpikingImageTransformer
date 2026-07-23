import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from tqdm import tqdm
import argparse
import os
import sys

from model import DSIT
from optimizer import ProximalAdamW


# ---------------------------------------------------------------------------
# MixUp + CutMix Regularization Utilities
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    B, C, H, W = x.shape
    index = torch.randperm(B, device=x.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy = torch.randint(H, (1,)).item()
    cx = torch.randint(W, (1,)).item()
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)

    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[index], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, scaler, device, accum_steps=4, use_amp=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # Augmentation: MixUp (33%), CutMix (33%), plain (34%)
        r = torch.rand(1).item()
        if r < 0.33:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=1.0)
            use_mixed = True
        elif r < 0.66:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            use_mixed = True
        else:
            use_mixed = False

        # Forward pass (AMP disabled — custom DAPSG autograd produces NaN in float16)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            if use_mixed:
                loss = mixed_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
            else:
                loss = criterion(logits, labels) / accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update Dopamine Tracker based on intrinsic entropy
        model.d_tracker.update(logits.detach())

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * accum_steps
        preds = logits.argmax(dim=1)
        if use_mixed:
            target_label = y_a if lam >= 0.5 else y_b
            correct += (preds == target_label).sum().item()
        else:
            correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f"{total_loss / (batch_idx + 1):.4f}",
            'acc': f"{100.0 * correct / total:.2f}%",
            'D(t)': f"{model.d_tracker.get_D():.4f}"
        })

    # Handle remaining gradients
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / len(dataloader), correct / total


# ---------------------------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


# ---------------------------------------------------------------------------
# Data Collation
# ---------------------------------------------------------------------------
def make_collate_fn(img_size=224):
    """Create a collate function that resizes images to the specified size."""
    if img_size <= 64:
        train_aug = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    else:
        train_aug = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    transform = train_aug

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def train_collate(batch):
        images, labels = [], []
        for item in batch:
            img = item.get('img', item.get('image'))
            label = item.get('fine_label', item.get('label'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(transform(img))
            labels.append(label)
        return {'image': torch.stack(images), 'label': torch.tensor(labels)}

    def val_collate(batch):
        images, labels = [], []
        for item in batch:
            img = item.get('img', item.get('image'))
            label = item.get('fine_label', item.get('label'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(val_transform(img))
            labels.append(label)
        return {'image': torch.stack(images), 'label': torch.tensor(labels)}

    return train_collate, val_collate


# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------
def log_pruning_status(optimizer, model, epoch):
    """Print how many attention heads are alive per block."""
    masks = optimizer.get_head_alive_mask(model)
    total_alive = 0
    total_heads = 0
    for i, m in enumerate(masks):
        alive = int(m.sum().item())
        total_alive += alive
        total_heads += len(m)
        print(f"  Block {i:2d}: {alive}/{len(m)} heads alive")
    print(f"  Total: {total_alive}/{total_heads} heads alive ({100.0 * total_alive / total_heads:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="D-SIT Training")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='HuggingFace dataset name (cifar100, imagenet-1k, frgfm/imagenette)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (default: 32 for cifar, 224 for imagenet)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per step')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension (512 for CIFAR 80%+ target, 768 for ImageNet)')
    parser.add_argument('--depth', type=int, default=12,
                        help='Number of transformer blocks (12 for 80%+ target, 8 for quick runs)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (must divide embed_dim)')
    parser.add_argument('--T', type=int, default=6,
                        help='Number of simulation timesteps (6 for 80%+ target, 4 for quick runs)')
    parser.add_argument('--prox_lambda', type=float, default=1e-4,
                        help='Proximal sparsity regularization strength')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable AMP mixed precision (BROKEN with custom DAPSG — use only for debugging)')
    args = parser.parse_args()

    # --- Dataset Configuration ---
    if args.dataset == 'cifar100':
        num_classes = 100
        img_size = args.img_size or 64
        train_split, val_split = 'train', 'test'
    elif 'imagenette' in args.dataset:
        num_classes = 10
        img_size = args.img_size or 224
        train_split, val_split = 'train', 'validation'
    else:
        num_classes = 1000
        img_size = args.img_size or 224
        train_split, val_split = 'train', 'validation'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data Loading ---
    print(f"Loading {args.dataset} (img_size={img_size})...")
    train_ds = load_dataset(args.dataset, split=train_split, trust_remote_code=True)
    val_ds = load_dataset(args.dataset, split=val_split, trust_remote_code=True)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_collate, val_collate = make_collate_fn(img_size)
    # Use num_workers=2 to avoid serialization issues on Colab/Windows
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate, num_workers=2, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=val_collate, num_workers=2, pin_memory=True)

    # --- Model ---
    print(f"Building D-SIT: embed_dim={args.embed_dim}, depth={args.depth}, "
          f"heads={args.num_heads}, T={args.T}, classes={num_classes}")
    model = DSIT(
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        T=args.T,
        img_size=img_size
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.2f}M")

    # --- Optimizer & Scheduler ---
    optimizer = ProximalAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-2,
        prox_lambda=args.prox_lambda,
        num_heads=args.num_heads
    )
    # Warmup + Cosine schedule: critical for SNN to stabilize firing rates early
    warmup_epochs = min(20, args.epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    # Change 8 (consensus): label smoothing 0.2 -> 0.1 for CIFAR-100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    print(f"AMP: {'ENABLED' if args.amp else 'DISABLED (safe mode for custom surrogate gradients)'}")

    start_epoch = 0
    best_val_acc = 0.0

    # --- Resume ---
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'D_t' in ckpt:
            model.d_tracker.current_D = ckpt['D_t']
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc*100:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}, D(t): {model.d_tracker.get_D():.6f}")

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.accum_steps, use_amp=args.amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")
        print(f"Dopamine D(t): {model.d_tracker.get_D():.6f}")

        # Log head pruning status every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Head Pruning Status:")
            log_pruning_status(optimizer, model, epoch)

        # Save best + latest
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'D_t': model.d_tracker.get_D(),
        }
        torch.save(ckpt, 'dsit_latest.pth')
        if is_best:
            torch.save(ckpt, 'dsit_best.pth')
            unique_name = f'dsit_best_epoch{epoch+1}_acc{best_val_acc * 100:.2f}.pth'
            torch.save(ckpt, unique_name)
            print(f"  *** New best val acc: {best_val_acc * 100:.2f}% (Saved as {unique_name}) ***")

    print(f"\nTraining complete. Best val acc: {best_val_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
