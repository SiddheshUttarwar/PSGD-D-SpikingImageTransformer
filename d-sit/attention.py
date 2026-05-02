import torch
import torch.nn as nn
from neurons import LIFNode


class ShiftedReLUSign(torch.autograd.Function):
    """Ternary activation {-1, 0, 1} for exploitation heads."""
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        ctx.save_for_backward(x)
        # torch.where keeps static shapes — boolean fancy-index assignment
        # (out[mask] = val) creates dynamic shapes that force XLA to recompile every step.
        return (x > threshold).to(x.dtype) - (x < -threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x.abs() <= 1.0).to(grad_output.dtype), None


class TernaryLIFNode(nn.Module):
    """LIF Node outputting ternary spikes {-1, 0, 1}."""
    def __init__(self, v_th=0.5, tau=4.0):
        super().__init__()
        self.v_th = v_th
        self.decay = 1.0 - (1.0 / tau)
        self.u = None

    def reset_state(self):
        self.u = None

    def forward(self, x):
        if self.u is None:
            self.u = torch.zeros_like(x)
        self.u = self.u.detach() * self.decay + x
        spike = ShiftedReLUSign.apply(self.u, self.v_th)
        self.u = self.u - spike.detach() * self.v_th
        return spike


class HeterogeneousSpikingSelfAttention(nn.Module):
    """
    Heterogeneous Spiking Self-Attention (SSA).

    Key fixes from diagnostic:
    - Attention output is normalized by sequence length N to prevent
      post-attention LIF saturation (was 100% firing → killed MLP).
    - BatchNorm before post-attention LIF for proper input distribution.
    - No Softmax (spike-driven), no fixed head_dim scaling.
    """
    def __init__(self, embed_dim=768, num_heads=12, num_binary_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_binary_heads = num_binary_heads
        self.num_ternary_heads = num_heads - num_binary_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        # Linear projections (targets for proximal pruning)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.weight.is_attn_head_weight = True

        # BatchNorm before Q, K, V LIF neurons
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)

        # LIF nodes for Q, K (binary)
        self.q_lif = LIFNode()
        self.k_lif = LIFNode()

        # Heterogeneous V
        self.v_lif_binary = LIFNode()
        self.v_lif_ternary = TernaryLIFNode()

        # BatchNorm BEFORE post-attention LIF to normalize attention output
        self.post_attn_bn = nn.BatchNorm1d(embed_dim)
        self.post_attn_lif = LIFNode(v_th=0.5)

    def reset_state(self):
        self.q_lif.reset_state()
        self.k_lif.reset_state()
        self.v_lif_binary.reset_state()
        self.v_lif_ternary.reset_state()
        self.post_attn_lif.reset_state()

    def forward(self, x, d_tracker, mask=None):
        B, N, D = x.shape

        q_c = self.q_bn(self.q_proj(x).reshape(B * N, -1)).reshape(B, N, -1)
        k_c = self.k_bn(self.k_proj(x).reshape(B * N, -1)).reshape(B, N, -1)
        v_c = self.v_bn(self.v_proj(x).reshape(B * N, -1)).reshape(B, N, -1)

        q_spike, _ = self.q_lif(q_c, d_tracker)
        k_spike, _ = self.k_lif(k_c, d_tracker)

        split_idx = self.num_binary_heads * self.head_dim
        v_spike_bin, _ = self.v_lif_binary(v_c[:, :, :split_idx], d_tracker)
        v_spike_ter = self.v_lif_ternary(v_c[:, :, split_idx:])
        v_spike = torch.cat([v_spike_bin, v_spike_ter], dim=-1)

        # Reshape for multi-head
        q = q_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Spike-driven attention with proper normalization
        # Q @ K^T gives integer counts (~5 per element), summing over N positions
        # would give ~300+ per element without normalization → saturates post-attn LIF.
        # Normalize by N to keep values in a reasonable range for the LIF threshold.
        scale = 1.0 / N
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        x_attn = attn @ v  # (B, H, N, d_h)

        if mask is not None:
            x_attn = x_attn * mask.view(1, self.num_heads, 1, 1)

        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)

        # Normalize before post-attention LIF to prevent saturation
        x_attn = self.post_attn_bn(x_attn.reshape(B * N, -1)).reshape(B, N, -1)
        attn_spike, _ = self.post_attn_lif(x_attn, d_tracker)
        attn_spike = self.dropout(attn_spike)

        return self.out_proj(attn_spike)
