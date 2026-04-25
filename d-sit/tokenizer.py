import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDelayTokenizer(nn.Module):
    """
    Learnable Time-Delay Tokenizer.
    Assigns a continuous learnable delay \Delta t_d to each of the D=768 dimensions.
    """
    def __init__(self, embed_dim=768, max_delay=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Initialize delays close to 0
        self.delays = nn.Parameter(torch.zeros(embed_dim))
        self.max_delay = max_delay

    def forward(self, x):
        """
        x: Spatiotemporal tensor of shape (T, B, N, D)
           T: time steps
           B: batch size
           N: sequence length (e.g. 196)
           D: embedding dim (e.g. 768)
        Returns delayed version of x.
        """
        T, B, N, D = x.shape
        
        # Constrain delays to be between 0 and max_delay
        dt = torch.clamp(self.delays, 0, self.max_delay) # Shape: (D,)
        
        # We will apply a soft temporal shift using linear interpolation.
        # For a delay dt[d], the value at time t is a blend of (t) and (t-1).
        # x_delayed(t) = (1 - dt) * x(t) + dt * x(t-1)
        # Assuming x(t<0) = 0
        
        # Create a shifted version of x
        # Pad time dimension at the beginning with 1 zero, and drop the last time step
        x_shifted = torch.cat([torch.zeros(1, B, N, D, device=x.device, dtype=x.dtype), x[:-1]], dim=0)
        
        # Reshape dt to broadcast over (T, B, N, D)
        dt_broadcast = dt.view(1, 1, 1, D)
        
        # Soft delay interpolation
        x_delayed = (1.0 - dt_broadcast) * x + dt_broadcast * x_shifted
        
        return x_delayed
