import torch
import torch.nn as nn
from neurons import LIFNode, TernaryLIFNode, DAPSG


class HeterogeneousSpikingSelfAttention(nn.Module):
    """
    Heterogeneous Spiking Self-Attention (SSA) — v3 (consensus-revised).

    Consensus fixes applied (3-agent debate, 3 rounds + design round):

    1. scale = 1/d_h (not 1/sqrt(d_h))
       Binary Q,K in {0,1} have E[q*k] = d*p^2 (not 0).
       Scaling by 1/sqrt(d) leaves mean = sqrt(d)*p^2 -> diverges with depth.
       1/d_h maps attention scores to approx [0,1] range. Biologically:
       divisive normalization via GABAergic interneuron networks.

    2. Learnable per-head attention scale (attn_scale)
       Applied before out_proj with no BN in between (BN scale-invariance
       would zero its gradient). out_proj mixes heads, so out_bn can't cancel.

    3. All-ternary V {-1, 0, +1} via TernaryDAPSG (Signed Binary Encoding)
       +1 = glutamatergic excitation, -1 = GABAergic inhibition (FFI motif).
       TernaryDAPSG piecewise backward eliminates ghost gradients at u=0.

    4. out_proj operates on float attn_out (pre-LIF)
       Full floating-point attended signal -> fine-grained projection learning.

    5. alpha_base=2.0, kappa=3.0, symmetric D(t) in [-0.3, 0.3].
    """
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        # Linear projections (targets for proximal pruning)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.weight.is_attn_head_weight = True

        # BatchNorm before Q, K, V spiking neurons
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)

        # Binary LIF for Q and K — spike-driven inner product
        self.q_lif = LIFNode()
        self.k_lif = LIFNode()

        # All-ternary V: {-1, 0, +1} with DA-PSG — richer value representation
        self.v_lif = TernaryLIFNode()

        # Learnable per-head output scale — applied in head space before out_proj.
        # IMPORTANT: must NOT have a BN between attn_scale and out_proj.
        # BN is scale-invariant: BN(s·x) = BN(x), so any BN after attn_scale would
        # produce an identically-zero gradient. By applying attn_scale directly into
        # out_proj (which mixes heads), out_bn (post-projection) cannot cancel it.
        self.attn_scale = nn.Parameter(torch.ones(num_heads))

        # BatchNorm AFTER out_proj — normalizes the mixed projection output
        self.out_bn = nn.BatchNorm1d(embed_dim)
        self.post_attn_lif = LIFNode(v_th=0.5)
        self.dropout = nn.Dropout(0.1)
        
        # DLIA (Dopaminergic Lateral Inhibitory Attention) Internal States
        self.U_attn = None
        self.A_attn = None
        self.v_th_attn = 0.5
        self.lambda_attn = 0.5

    def reset_state(self):
        self.q_lif.reset_state()
        self.k_lif.reset_state()
        self.v_lif.reset_state()
        self.post_attn_lif.reset_state()
        self.U_attn = None
        self.A_attn = None

    def forward(self, x, d_tracker, mask=None, D_local=None):
        B, N, D = x.shape

        # Project + BatchNorm + spike for Q, K, V
        q_c = self.q_bn(self.q_proj(x).reshape(B * N, -1)).reshape(B, N, -1)
        k_c = self.k_bn(self.k_proj(x).reshape(B * N, -1)).reshape(B, N, -1)
        v_c = self.v_bn(self.v_proj(x).reshape(B * N, -1)).reshape(B, N, -1)

        q_spike, _ = self.q_lif(q_c, d_tracker, D_local=D_local)           # {0, 1}
        k_spike, _ = self.k_lif(k_c, d_tracker, D_local=D_local)           # {0, 1}
        v_spike = self.v_lif(v_c, d_tracker, D_local=D_local)              # {-1, 0, +1}

        # Reshape for multi-head attention
        q = q_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_spike.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # DLIA: Compute raw overlap M_ij
        scale = 1.0 / self.head_dim
        M = (q @ k.transpose(-2, -1)) * scale   # (B, H, N, N)

        if mask is not None:
            M = M * mask.view(1, self.num_heads, 1, 1)
            
        # DLIA: Initialize recurrent states
        if self.U_attn is None:
            self.U_attn = torch.zeros_like(M)
            self.A_attn = torch.zeros_like(M)
            
        # DLIA: Compute lateral inhibition
        row_sum = self.A_attn.sum(dim=-1, keepdim=True)
        lat_inh = row_sum - self.A_attn
        
        # DLIA: Localized Dopaminergic Gating
        if D_local is not None:
            # D_local shape (B, N, 1) -> broadcast to (B, 1, N, 1) for (B, H, N, N) attention
            D_mod = D_local.view(B, 1, N, 1)
        else:
            D_mod = torch.tensor(d_tracker.get_D(), dtype=x.dtype, device=x.device)
            
        # D > 0 (exploit) -> high gamma (sparse attention)
        # D < 0 (explore) -> low gamma (diffuse attention)
        gamma = 0.5 * (1.0 + 3.0 * D_mod).clamp(min=0.0)
        
        # DLIA: Update Interneuron Membrane Potential
        self.U_attn = self.lambda_attn * self.U_attn + M - gamma * lat_inh - self.v_th_attn * self.A_attn
        
        # DLIA: Spiking Softmax via DAPSG
        v_th_tensor = torch.tensor(self.v_th_attn, device=x.device, dtype=x.dtype)
        # D_local is (B, N, 1). To apply it over the (B, H, N, N) attention matrix,
        # we must broadcast it to match U_attn.
        if D_local is not None:
            D_local_broadcast = D_local.view(B, 1, N, 1).expand(-1, self.num_heads, -1, N)
        else:
            D_local_broadcast = None
        self.A_attn = DAPSG.apply(self.U_attn, v_th_tensor, d_tracker, 2.0, 3.0, D_local_broadcast)
        
        # Route Information via Spiking Softmax
        x_attn = self.A_attn @ v   # (B, H, N, d_h)

        # Fix 2: learnable per-head output scale in head space, before out_proj.
        # No BN between attn_scale and out_proj — BN is scale-invariant (BN(s·x)=BN(x))
        # which would zero the gradient. out_proj mixes heads so out_bn cannot cancel it.
        x_attn = x_attn * self.attn_scale.view(1, self.num_heads, 1, 1)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)

        # Fix 4: out_proj on float representation (not binary spikes)
        # Projection sees the full floating-point attended signal → learns fine-grained transforms
        x_out = self.out_proj(x_attn)

        # Normalize after projection, then spike
        x_out = self.out_bn(x_out.reshape(B * N, -1)).reshape(B, N, -1)
        attn_spike, _ = self.post_attn_lif(x_out, d_tracker, D_local=D_local)
        attn_spike = self.dropout(attn_spike)

        # Detach temporal DLIA memory states to prevent O(T) BPTT graph buildup
        self.U_attn = self.U_attn.detach()
        self.A_attn = self.A_attn.detach()

        return attn_spike
