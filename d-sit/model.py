import torch
import torch.nn as nn
from neurons import LIFNode, IntrinsicDopamineTracker
from tokenizer import TimeDelayTokenizer
from attention import HeterogeneousSpikingSelfAttention
from astrocyte import AstrocyticSyncytium


class HybridANNConvStem(nn.Module):
    """
    Hybrid ANN Dense Stem for >80% CIFAR-100 Accuracy.
    
    Instead of aggressively downsampling 32x32 to 8x8 (64 tokens) using spikes, 
    this continuous stem extracts rich continuous features and outputs a 16x16 
    (256 token) dense grid. This dramatically preserves spatial geometry.
    """
    def __init__(self, in_channels=3, embed_dim=768, img_size=32):
        super().__init__()
        self.img_size = img_size
        
        # 32x32 -> stride 2 -> 16x16 (dense grid)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()
        
        # 16x16 -> stride 1 -> 16x16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.act3 = nn.GELU()
        
    def reset_state(self):
        # Continuous stem has no spiking state
        pass

    def forward(self, x, d_tracker=None):
        """
        x: (B, C, H, W)
        Returns: (B, N, D) continuous floats
        """
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        # Flatten spatial -> (B, N, D)
        out = x.flatten(2).transpose(1, 2)
        return out


class SpikingMLP(nn.Module):
    """
    Spiking MLP with 4x expansion (D -> 4D -> D).
    Supports membrane shortcuts from the previous transformer block.
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = LIFNode()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.lif2 = LIFNode()
        self.dropout = nn.Dropout(0.1)

    def reset_state(self):
        self.lif1.reset_state()
        self.lif2.reset_state()

    def forward(self, x, d_tracker, residual_u=None, D_local=None):
        B, N, D = x.shape

        c1 = self.fc1(x)
        c1 = self.bn1(c1.reshape(B * N, -1)).reshape(B, N, -1)
        s1, _ = self.lif1(c1, d_tracker, D_local=D_local)
        s1 = self.dropout(s1)

        c2 = self.fc2(s1)
        c2 = self.bn2(c2.reshape(B * N, -1)).reshape(B, N, -1)
        s2, u2 = self.lif2(c2, d_tracker, residual_u=residual_u, D_local=D_local)

        return s2, u2


class DSITBlock(nn.Module):
    """
    Single D-SIT transformer block: SSA -> S-MLP with dual ReZero residuals.

    Consensus changes (3-agent debate):
    - Change 6: Two separate learnable res_scale params (attn + MLP).
      Attention output (spikes) and MLP output (floats) have different statistics,
      requiring independent geometric scaling.
    - Change 6: Attention NOW has its own residual: x = x + res_scale_attn * Attn(x).
      Without this, attention had no gradient bypass in 12-block networks.
    - Change 7: Both init at 0 (true ReZero). Identity mapping guaranteed at step 0.
      First gradient: dL/d_scale = branch_output * upstream_grad != 0 (non-zero spikes).
      Biologically: silent synapses (NMDA-only) that mature via activity-dependent LTP.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.attn = HeterogeneousSpikingSelfAttention(embed_dim, num_heads)
        self.mlp = SpikingMLP(embed_dim, int(embed_dim * mlp_ratio))
        # Change 7: init=0 for true ReZero (identity mapping at step 0)
        self.res_scale_attn = nn.Parameter(torch.zeros(1))
        self.res_scale_mlp  = nn.Parameter(torch.zeros(1))

    def reset_state(self):
        self.attn.reset_state()
        self.mlp.reset_state()

    def forward(self, x, d_tracker, mask=None, residual_u=None, D_local=None):
        # Change 6: Dual residual — attention gets its own identity bypass.
        # Without this, gradient through attention in 12-block network had no shortcut.
        x_attn = self.attn(x, d_tracker, mask=mask, D_local=D_local)
        x = x + self.res_scale_attn * x_attn          # attention residual

        # MLP receives post-attention-residual x (not raw x_attn)
        x_out, u_out = self.mlp(x, d_tracker, residual_u=residual_u, D_local=D_local)
        x = x + self.res_scale_mlp * x_out            # MLP residual

        return x, u_out


class DSIT(nn.Module):
    """
    Dopaminergic Spiking Image Transformer (D-SIT).

    Full pipeline:
      1. Static image -> T timestep repetition
      2. Spiking Conv Stem -> spatial tokens
      3. Learnable Time-Delay Tokenizer
      4. L transformer blocks with heterogeneous SSA + S-MLP + residual
      5. Spatio-Temporal Average Pooling
      6. Linear classifier
    """
    def __init__(self, in_channels=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, T=4, img_size=224):
        super().__init__()
        self.T = T
        self.depth = depth
        self.stem = HybridANNConvStem(in_channels, embed_dim, img_size=img_size)
        self.tokenizer = TimeDelayTokenizer(embed_dim)
        
        # Instantiate Astrocytic Syncytium for localized credit assignment
        self.astrocyte = AstrocyticSyncytium()

        # Learnable 2D positional embedding — tokens are otherwise spatially unaware
        if img_size >= 224:
            num_patches = (img_size // 16) ** 2
        else:
            num_patches = (img_size // 2) ** 2  # Dense Tokenizer (stride 2 total -> 16x16 = 256 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DSITBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.head = nn.Linear(embed_dim, num_classes)
        self.d_tracker = IntrinsicDopamineTracker()

        # Initialize weights for spiking-friendly distributions
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce healthy firing rates at initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Slightly larger init to push membrane potentials past threshold
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    # Positive bias shift helps neurons fire (BN output > 0 more often)
                    nn.init.constant_(m.bias, 0.1)

    def reset_all_states(self):
        """Reset membrane potentials of ALL spiking neurons in the network."""
        self.stem.reset_state()
        self.astrocyte.reset_state()
        for blk in self.blocks:
            blk.reset_state()

    def forward(self, x):
        """
        x: (B, C, H, W) — a single static image batch
        Returns: (B, num_classes) logits
        """
        self.reset_all_states()

        # 1. Stem: process the same image over T timesteps
        stem_outputs = []
        for t in range(self.T):
            s_out = self.stem(x, self.d_tracker)
            stem_outputs.append(s_out)

        x_seq = torch.stack(stem_outputs, dim=0)  # (T, B, N, D)

        # 2. Learnable Time-Delay Tokenizer
        x_seq = self.tokenizer(x_seq)

        # 3. Add positional embedding (broadcast over T and B dimensions)
        x_seq = x_seq + self.pos_embed.unsqueeze(0)  # (1,1,N,D) broadcasts over (T,B,N,D)

        # 4. Transformer Blocks
        for blk in self.blocks:
            blk.reset_state()
            block_outputs = []
            u_prev = None
            for t in range(self.T):
                x_t = x_seq[t]
                
                # Astrocytic Integration: Compute localized dopamine based on activity
                D_local = self.astrocyte(x_t, self.d_tracker.get_D())
                
                x_t, u_prev = blk(x_t, self.d_tracker, mask=None, residual_u=u_prev, D_local=D_local)
                block_outputs.append(x_t)
            x_seq = torch.stack(block_outputs, dim=0)

        # 5. Spatio-Temporal Average Pooling: (T, B, N, D) -> (B, D)
        x_pool = x_seq.mean(dim=(0, 2))

        # 6. Classifier
        logits = self.head(x_pool)
        return logits
