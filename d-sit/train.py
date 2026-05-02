import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import argparse
import contextlib
import os
import sys

from model import DSIT
from optimizer import ProximalAdamW


# ---------------------------------------------------------------------------
# TPU / GPU / CPU backend detection
# ---------------------------------------------------------------------------
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False


def _get_device():
    """Auto-detect TPU → GPU → CPU in that priority order."""
    if _HAS_XLA:
        try:
            import torch_xla
            device = torch_xla.device()
            return device, 'tpu'
        except Exception as e:
            print(f"[warn] TPU unavailable ({e}), falling back to GPU/CPU.")
    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    return torch.device('cpu'), 'cpu'


def _save(obj, path, device_type='cuda'):
    """Checkpoint save: xm.save on TPU (ensures lazy tensors materialise), torch.save elsewhere."""
    if device_type == 'tpu':
        xm.save(obj, path)
    else:
        torch.save(obj, path)


class _NoOpScaler:
    """Mimics GradScaler's interface for TPU/CPU — no-op passthrough."""
    def scale(self, loss): return loss
    def unscale_(self, opt): pass
    def step(self, opt): opt.step()
    def update(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, _): pass


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
def train_epoch(model, dataloader, optimizer, criterion, scaler,
                device, device_type='cuda', accum_steps=4, use_amp=False, verbose=True):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    # TPU: MpDeviceLoader prefetches batches onto TPU asynchronously
    loader = pl.MpDeviceLoader(dataloader, device) if device_type == 'tpu' else dataloader

    # autocast is CUDA-only; use nullcontext on TPU/CPU
    autocast_ctx = (
        torch.amp.autocast('cuda', enabled=use_amp)
        if device_type == 'cuda'
        else contextlib.nullcontext()
    )

    non_blocking = (device_type == 'cuda')
    pbar = tqdm(loader, desc="Training", disable=not verbose)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=non_blocking)
        labels = batch['label'].to(device, non_blocking=non_blocking)

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

        with autocast_ctx:
            logits = model(images)
            if use_mixed:
                loss = mixed_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
            else:
                loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        model.d_tracker.update(logits.detach())

        if (batch_idx + 1) % accum_steps == 0:
            if device_type == 'tpu':
                # On TPU: clip grads, then xm.optimizer_step (calls mark_step internally)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                xm.optimizer_step(optimizer)
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()

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

    # Flush any remaining accumulated gradients
    if (batch_idx + 1) % accum_steps != 0:
        if device_type == 'tpu':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)
        else:
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
def validate(model, dataloader, criterion, device, device_type='cuda', verbose=True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    loader = pl.MpDeviceLoader(dataloader, device) if device_type == 'tpu' else dataloader
    non_blocking = (device_type == 'cuda')

    for batch in tqdm(loader, desc="Validating", disable=not verbose):
        images = batch['image'].to(device, non_blocking=non_blocking)
        labels = batch['label'].to(device, non_blocking=non_blocking)

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
            images.append(train_aug(img))
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
def _parse_args():
    parser = argparse.ArgumentParser(description="D-SIT Training")
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--T', type=int, default=6)
    parser.add_argument('--prox_lambda', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable AMP (CUDA only; BROKEN with custom DAPSG)')
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = _parse_args()

    # --- Device ---
    device, device_type = _get_device()

    # Multi-core TPU: each xmp.spawn worker gets its own ordinal (core index)
    rank = xm.get_ordinal() if device_type == 'tpu' else 0
    world_size = xm.xrt_world_size() if device_type == 'tpu' else 1
    is_master = (rank == 0)

    def log(*msgs):
        if is_master:
            print(*msgs)

    log(f"Device: {device_type.upper()} ({device})")
    if device_type == 'cuda':
        log(f"GPU:  {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device_type == 'tpu':
        import torch_xla
        log(f"torch_xla: {torch_xla.__version__}, world_size: {world_size}")

    use_amp = args.amp and (device_type == 'cuda')

    # --- Dataset ---
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

    log(f"Loading {args.dataset} (img_size={img_size})...")
    train_ds = load_dataset(args.dataset, split=train_split, trust_remote_code=True)
    val_ds = load_dataset(args.dataset, split=val_split, trust_remote_code=True)
    log(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_collate, val_collate = make_collate_fn(img_size)
    pin = (device_type == 'cuda')

    # Multi-core TPU: shard the training set across cores via DistributedSampler
    if device_type == 'tpu' and world_size > 1:
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                  collate_fn=train_collate, num_workers=2,
                                  pin_memory=False, drop_last=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_collate, num_workers=2,
                                  pin_memory=pin, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=val_collate, num_workers=2,
                            pin_memory=pin, drop_last=True)

    # --- Model ---
    log(f"Building D-SIT: embed_dim={args.embed_dim}, depth={args.depth}, "
        f"heads={args.num_heads}, T={args.T}, classes={num_classes}")
    model = DSIT(
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        T=args.T,
        img_size=img_size
    ).to(device)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # --- Optimizer & Scheduler ---
    optimizer = ProximalAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-2,
        prox_lambda=args.prox_lambda,
        num_heads=args.num_heads
    )
    warmup_epochs = min(20, args.epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if device_type == 'cuda' else _NoOpScaler()
    log(f"AMP: {'ENABLED' if use_amp else 'DISABLED'}")

    start_epoch = 0
    best_val_acc = 0.0

    # --- Resume ---
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt and device_type == 'cuda':
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'D_t' in ckpt:
            model.d_tracker.current_D = ckpt['D_t']
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        log(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc*100:.2f}%")
        log(f"  LR: {scheduler.get_last_lr()[0]:.6f}, D(t): {model.d_tracker.get_D():.6f}")

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        # Shuffle reproducibly per epoch when using DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        log(f"\n{'='*60}")
        log(f"Epoch {epoch + 1}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.6f}")
        log(f"{'='*60}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, device_type, args.accum_steps, use_amp, verbose=is_master
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, device_type, verbose=is_master
        )

        scheduler.step()

        log(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        log(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")
        log(f"Dopamine D(t): {model.d_tracker.get_D():.6f}")

        if (epoch + 1) % 5 == 0 and is_master:
            log("Head Pruning Status:")
            log_pruning_status(optimizer, model, epoch)

        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        # Only master saves checkpoints to avoid 8 concurrent writes on TPU
        if is_master:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'D_t': model.d_tracker.get_D(),
            }
            _save(ckpt, 'dsit_latest.pth', device_type)
            if is_best:
                _save(ckpt, 'dsit_best.pth', device_type)
                unique_name = f'dsit_best_epoch{epoch+1}_acc{best_val_acc * 100:.2f}.pth'
                _save(ckpt, unique_name, device_type)
                log(f"  *** New best val acc: {best_val_acc * 100:.2f}% (Saved as {unique_name}) ***")

        # Keep all cores in lockstep — prevents a fast core from starting the next
        # epoch while another is still in validation or the save above.
        if device_type == 'tpu':
            xm.rendezvous(f'epoch_{epoch}_end')

    log(f"\nTraining complete. Best val acc: {best_val_acc * 100:.2f}%")


if __name__ == '__main__':
    # On Colab/single-host TPU (PJRT runtime), xmp.spawn causes init errors —
    # the PJRT runtime handles multi-core internally via xm.mark_step() and
    # xm.optimizer_step().  Just call main() directly for all backends.
    main()
