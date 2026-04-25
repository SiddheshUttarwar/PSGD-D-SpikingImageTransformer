import torch
import torch.nn as nn
from neurons import LIFNode


class ShiftedReLUSign(torch.autograd.Function):
    """
    Ternary activation function {-1, 0, 1} for exploitation heads.
    Forward: quantizes to {-1, 0, 1} based on threshold.
    Backward: Straight-Through Estimator clamped to [-1, 1].
    """
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        ctx.save_for_backward(x)
        out = torch.zeros_like(x)
        out[x > threshold] = 1.0
        out[x < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # STE: pass gradients through where |x| <= 1.0
        grad_x = grad_output.clone()
        grad_x[x.abs() > 1.0] = 0.0
        return grad_x, None


class TernaryLIFNode(nn.Module):
    """
    LIF Node that outputs ternary spikes {-1, 0, 1}.
    Used for the exploitation heads (heads 9-12) in the heterogeneous attention.
    """
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

    Key design choices for spike-driven attention:
    - Q, K: Binary {0, 1} through LIF neurons
    - V (heads 1-8): Binary {0, 1} — exploration heads
    - V (heads 9-12): Ternary {-1, 0, 1} — exploitation heads
    - NO Softmax, NO scaling (unlike ANN transformers)
      Scaling by head_dim^{-0.5} is for softmax stability — in spike-driven
      attention it crushes the output magnitude below the LIF threshold,
      killing post-attention neurons. We follow Spikformer and remove it.
    - Supports a pruning mask from the proximal optimizer.
    """
    def __init__(self, embed_dim=768, num_heads=12, num_binary_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_binary_heads = num_binary_heads
        self.num_ternary_heads = num_heads - num_binary_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Mark for proximal optimizer
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.weight.is_attn_head_weight = True

        # BatchNorm before Q, K, V LIF neurons to control input distribution
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)

        # LIF nodes
        self.q_lif = LIFNode()
        self.k_lif = LIFNode()
        self.v_lif_binary = LIFNode()
        self.v_lif_ternary = TernaryLIFNode()

        # Post-attention: use a lower threshold since attention outputs are sparse
        self.post_attn_lif = LIFNode(v_th=0.25)

        # Learnable scaling for attention output (replaces fixed head_dim^{-0.5})
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.5)

    def reset_state(self):
        self.q_lif.reset_state()
        self.k_lif.reset_state()
        self.v_lif_binary.reset_state()
        self.v_lif_ternary.reset_state()
        self.post_attn_lif.reset_state()

    def forward(self, x, d_tracker, mask=None):
        """
        x: (B, N, D) input at a single timestep
        Returns: (B, N, D)
        """
        B, N, D = x.shape

        q_c = self.q_bn(self.q_proj(x).reshape(B * N, -1)).reshape(B, N, -1)
        k_c = self.k_bn(self.k_proj(x).reshape(B * N, -1)).reshape(B, N, -1)
        v_c = self.v_bn(self.v_proj(x).reshape(B * N, -1)).reshape(B, N, -1)

        # Binary spikes for Q and K
        q_spike, _ = self.q_lif(q_c, d_tracker)
        k_spike, _ = self.k_lif(k_c, d_tracker)

        # Split V into binary (exploration) and ternary (exploitation)
        split_idx = self.num_binary_heads * self.head_dim
        v_spike_bin, _ = self.v_lif_binary(v_c[:, :, :split_idx], d_tracker)
        v_spike_ter = self.v_lif_ternary(v_c[:, :, split_idx:])
        v_spike = torch.cat([v_spike_bin, v_spike_ter], dim=-1)

        # Reshape for multi-head: (B, N, D) -> (B, H, N, d_h)
        q = q_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Spike-driven attention: NO softmax, NO fixed scaling
        # Q and K are binary -> Q @ K^T counts co-active dimensions (integer values)
        # Learnable scale replaces the fixed head_dim^{-0.5} that kills LIF neurons
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, H, N, N)
        x_attn = attn @ v  # (B, H, N, d_h)

        # Dynamic head pruning mask
        if mask is not None:
            x_attn = x_attn * mask.view(1, self.num_heads, 1, 1)

        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)

        # Post-attention LIF (low threshold to not kill signal)
        attn_spike, _ = self.post_attn_lif(x_attn, d_tracker)

        return self.out_proj(attn_spike)
