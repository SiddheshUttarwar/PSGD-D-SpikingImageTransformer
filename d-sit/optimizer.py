import torch
from torch.optim import AdamW


class ProximalAdamW(AdamW):
    """
    AdamW with a Proximal Gradient Descent step for l_{2,1}-norm group sparsity.

    After the standard AdamW update, applies soft-thresholding ONLY to the
    attention head projection weights (Q, K, V, O) that are explicitly marked
    with `is_attn_head_weight = True`.  This avoids accidentally pruning the
    classifier head, MLP weights, or convolutional stem.

    The proximal operator for each head group g is:
        W_g = max(1 - eta*lambda / ||W_g||_2, 0) * W_g
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, prox_lambda=1e-4, num_heads=12):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        self.prox_lambda = prox_lambda
        self.num_heads = num_heads

    @torch.no_grad()
    def step(self, closure=None):
        """Standard AdamW step followed by proximal operator on attention heads."""
        loss = super().step(closure)

        for group in self.param_groups:
            lr = group['lr']
            threshold = lr * self.prox_lambda

            for p in group['params']:
                if p.grad is None:
                    continue

                # Only apply to weights explicitly marked as attention head weights
                if not getattr(p, 'is_attn_head_weight', False):
                    continue

                if p.dim() != 2:
                    continue

                out_features, in_features = p.shape
                if out_features % self.num_heads != 0:
                    continue

                # Reshape to (num_heads, head_dim * in_features)
                w_reshaped = p.view(self.num_heads, -1)

                # L2 norm per head group
                norms = torch.norm(w_reshaped, p=2, dim=1, keepdim=True)  # (H, 1)

                # Soft-thresholding: max(1 - threshold/||W||, 0)
                scaling = torch.clamp(1.0 - threshold / (norms + 1e-8), min=0.0)

                p.copy_((w_reshaped * scaling).view(out_features, in_features))

        return loss

    def get_head_alive_mask(self, model):
        """
        Inspect the model's attention heads and return a binary mask indicating
        which heads are still alive (non-zero weight norms).
        Useful for logging pruning progress.
        """
        masks = []
        for blk in model.blocks:
            attn = blk.attn
            # Use output projection as the canonical indicator
            w = attn.out_proj.weight  # (D, D)
            w_reshaped = w.view(self.num_heads, -1)
            norms = torch.norm(w_reshaped, p=2, dim=1)
            alive = (norms > 1e-6).float()
            masks.append(alive)
        return masks
