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
# Training Loop
# ---------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, scaler, device, accum_steps=4):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, labels) / accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update Dopamine Tracker based on predictions
        model.d_tracker.update(labels, logits.detach())

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * accum_steps
        preds = logits.argmax(dim=1)
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

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4) if img_size <= 64 else transforms.RandomResizedCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per step')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension (256 for CIFAR T4, 768 for ImageNet)')
    parser.add_argument('--depth', type=int, default=8,
                        help='Number of transformer blocks (8 for CIFAR T4, 12 for ImageNet)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads (must divide embed_dim)')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of simulation timesteps')
    parser.add_argument('--prox_lambda', type=float, default=1e-4,
                        help='Proximal sparsity regularization strength')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # --- Dataset Configuration ---
    if args.dataset == 'cifar100':
        num_classes = 100
        img_size = args.img_size or 32
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
        weight_decay=5e-4,
        prox_lambda=args.prox_lambda,
        num_heads=args.num_heads
    )
    # Warmup + Cosine schedule: critical for SNN to stabilize firing rates early
    warmup_epochs = min(10, args.epochs // 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

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
            model, train_loader, optimizer, criterion, scaler, device, args.accum_steps
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
            print(f"  *** New best val acc: {best_val_acc * 100:.2f}% ***")

    print(f"\nTraining complete. Best val acc: {best_val_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
