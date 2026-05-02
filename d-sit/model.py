import torch
import torch.nn as nn
from neurons import LIFNode, IntrinsicDopamineTracker
from tokenizer import TimeDelayTokenizer
from attention import HeterogeneousSpikingSelfAttention


class SpikingConvStem(nn.Module):
    """
    4-Layer Spiking Convolutional Stem.

    For CIFAR-100 (32x32 input):
      Input 32x32  -> Layer1 (stride 2 + pool 2)  -> 8x8
                   -> Layer2-4 (stride 1)          -> 8x8
    Output: (B, 64, embed_dim) per timestep.

    For ImageNet (224x224 input):
      Input 224x224 -> Layer1 (stride 2 + pool 2) -> 56x56
                    -> Layer2 (stride 2)           -> 28x28
                    -> Layer3 (stride 2)           -> 14x14
                    -> Layer4 (stride 1)           -> 14x14
    Output: (B, 196, embed_dim) per timestep.
    """
    def __init__(self, in_channels=3, embed_dim=768, img_size=224):
        super().__init__()
        self.img_size = img_size

        if img_size >= 224:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.pool_shortcut = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.lif1 = LIFNode()
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.lif2 = LIFNode()

            self.conv3 = nn.Conv2d(128, 384, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(384)
            self.lif3 = LIFNode()

            self.conv4 = nn.Conv2d(384, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(embed_dim)
            self.lif4 = LIFNode()
        else:
            # CIFAR stem (32x32) -> 8x8 = 64 tokens
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.pool_shortcut = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.lif1 = LIFNode()
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.lif2 = LIFNode()

            self.conv3 = nn.Conv2d(128, 384, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(384)
            self.lif3 = LIFNode()

            self.conv4 = nn.Conv2d(384, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(embed_dim)
            self.lif4 = LIFNode()

    def reset_state(self):
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif3.reset_state()
        self.lif4.reset_state()

    def forward(self, x, d_tracker):
        """
        x: (B, C, H, W)
        Returns: (B, N, D) at a single timestep
        """
        x_pool = self.pool_shortcut(x)
        x = self.bn1(self.conv1(x) + x_pool)
        s1, _ = self.lif1(x, d_tracker)
        p1 = self.pool1(s1)

        x = self.bn2(self.conv2(p1))
        s2, _ = self.lif2(x, d_tracker)

        x = self.bn3(self.conv3(s2))
        s3, _ = self.lif3(x, d_tracker)

        x = self.bn4(self.conv4(s3))
        s4, _ = self.lif4(x, d_tracker)

        # Flatten spatial -> (B, N, D)
        out = s4.flatten(2).transpose(1, 2)
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

    def forward(self, x, d_tracker, residual_u=None):
        B, N, D = x.shape

        c1 = self.fc1(x)
        c1 = self.bn1(c1.reshape(B * N, -1)).reshape(B, N, -1)
        s1, _ = self.lif1(c1, d_tracker)
        s1 = self.dropout(s1)

        c2 = self.fc2(s1)
        c2 = self.bn2(c2.reshape(B * N, -1)).reshape(B, N, -1)
        s2, u2 = self.lif2(c2, d_tracker, residual_u=residual_u)

        return s2, u2


class DSITBlock(nn.Module):
    """
    Single D-SIT transformer block: SSA -> S-MLP
    with ReZero-style scaled residual to prevent gradient explosion.

    Diagnostic showed gradients exploding ~1000x per block through unscaled
    residual. ReZero initializes the residual scale at 0.1 so each block's
    contribution starts small and is learned during training.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.attn = HeterogeneousSpikingSelfAttention(embed_dim, num_heads)
        self.mlp = SpikingMLP(embed_dim, int(embed_dim * mlp_ratio))
        # ReZero: learnable scale, 0.1 lets blocks contribute meaningfully while still stable
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def reset_state(self):
        self.attn.reset_state()
        self.mlp.reset_state()

    def forward(self, x, d_tracker, mask=None, residual_u=None):
        identity = x

        # Attention
        x_attn = self.attn(x, d_tracker, mask=mask)

        # MLP with membrane shortcut
        x_out, u_out = self.mlp(x_attn, d_tracker, residual_u=residual_u)

        # Scaled residual: prevents gradient explosion through the block chain
        x_out = identity + self.res_scale * x_out

        return x_out, u_out


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
        self.stem = SpikingConvStem(in_channels, embed_dim, img_size=img_size)
        self.tokenizer = TimeDelayTokenizer(embed_dim)

        # Learnable 2D positional embedding — tokens are otherwise spatially unaware
        if img_size >= 224:
            num_patches = (img_size // 16) ** 2
        else:
            num_patches = (img_size // 4) ** 2
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
                x_t, u_prev = blk(x_t, self.d_tracker, mask=None, residual_u=u_prev)
                block_outputs.append(x_t)
            x_seq = torch.stack(block_outputs, dim=0)

        # 5. Spatio-Temporal Average Pooling: (T, B, N, D) -> (B, D)
        x_pool = x_seq.mean(dim=(0, 2))

        # 6. Classifier
        logits = self.head(x_pool)
        return logits
