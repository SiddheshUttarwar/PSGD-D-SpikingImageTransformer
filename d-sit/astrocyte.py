import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AstrocyticSyncytium(nn.Module):
    """
    Simulates the continuous Calcium (Ca2+) diffusion network of Astrocytes.
    Provides localized credit assignment (O(N) spatial routing) by scaling the
    global Dopamine signal D(t) with local Ca2+ concentration.
    
    This turns the scalar global reward into a dense spatial gradient field.
    """
    def __init__(self, beta=0.9, kappa=0.1):
        super().__init__()
        self.beta = beta
        self.kappa = kappa
        
        # 2D spatial Laplacian diffusion kernel (3x3)
        # Models the physical diffusion of Ca2+ to neighboring astrocytes
        kernel = torch.tensor([
            [0.5, 1.0, 0.5],
            [1.0, -6.0, 1.0],
            [0.5, 1.0, 0.5]
        ])
        kernel = kernel / 6.0  # Normalize
        self.register_buffer('laplacian', kernel.view(1, 1, 3, 3))
        self.C = None
        
    def reset_state(self):
        self.C = None
        
    def forward(self, activity, D_global):
        """
        activity: Local neural activity (B, N, D) or (B, N)
        D_global: float scalar Dopamine RPE
        Returns: D_local (B, N, 1) localized dopamine modulation
        """
        B, N = activity.shape[:2]
        grid_size = int(math.sqrt(N))
        assert grid_size * grid_size == N, "N must be a perfect square for spatial diffusion."
        
        if self.C is None:
            self.C = torch.zeros(B, 1, grid_size, grid_size, 
                                 device=activity.device, dtype=torch.float32)
                                 
        # Average activity across feature channels
        if activity.dim() == 3:
            local_activity = activity.abs().mean(dim=-1)
        else:
            local_activity = activity.abs()
            
        local_activity = local_activity.view(B, 1, grid_size, grid_size)
        
        # PDE Integration: Compute diffusion
        diffused = F.conv2d(self.C, self.laplacian, padding=1)
        
        # Update Calcium state: C(t) = decay + input + diffusion
        self.C = self.beta * self.C + local_activity + self.kappa * diffused
        
        # Normalize Calcium state to [0, 1] using Sigmoid to prevent explosion
        # Shifted slightly to keep resting Ca2+ near 0.2 - 0.3
        C_norm = torch.sigmoid(self.C - 1.0)
        
        # Localize Dopamine: scale global scalar by local Ca2+ concentration
        C_flat = C_norm.view(B, N, 1)
        D_local = D_global * C_flat
        
        return D_local
