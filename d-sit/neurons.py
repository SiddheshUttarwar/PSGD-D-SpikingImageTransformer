import torch
import torch.nn as nn


class IntrinsicDopamineTracker:
    """
    Autonomous Dopamine generator using Shannon Entropy to track the
    network's Intrinsic Reward Prediction Error (RPE).
    
    H[t] = - sum(P_c * log(P_c))
    R[t] = -(H[t] - H[t-1])
    V[t] = beta * V[t-1] + (1 - beta) * R[t]
    D[t] = R[t] - V[t-1]
    """
    def __init__(self, beta=0.7):
        self.beta = beta
        self.V_t = 0.0
        self.prev_H = None
        self.current_D = 0.0

    def update(self, logits: torch.Tensor):
        """
        Calculates dopamine D[t] based on the entropy of the current prediction.
        Args:
            logits: Output logits of the classification head.
        """
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            # H[t] = - sum(P * log(P))
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
            
            if self.prev_H is None:
                self.prev_H = entropy
                self.current_D = 0.0
                return
                
            # Intrinsic reward R[t] = -(H[t] - H[t-1])
            R_t = -(entropy - self.prev_H)
            self.prev_H = entropy
            
            # Dopamine D[t] = R[t] - V[t-1]
            self.current_D = R_t - self.V_t
            
            # Update expected baseline V[t]
            self.V_t = self.beta * self.V_t + (1.0 - self.beta) * R_t

    def get_D(self):
        # Clip to [0, 0.3]: values above this consistently cause catastrophic val drops
        # (observed: D>0.5 correlates with 8-13% accuracy crashes due to DAPSG alpha collapse)
        return max(0.0, min(self.current_D, 0.3))


class DAPSG(torch.autograd.Function):
    """
    Dopamine-Modulated Proximal Surrogate Gradient (DA-PSG).

    Forward: Heaviside step function s = Theta(u - V_th)
    Backward: sigma'_DA(u) = 1/(2*alpha) * (1 + |u - V_th|/alpha)^(-2)
              where alpha = alpha_base / (1 + kappa * D(t))
    """
    @staticmethod
    def forward(ctx, u, v_th, d_tracker, alpha_base, kappa):
        ctx.save_for_backward(u, v_th)
        ctx.d_tracker = d_tracker
        ctx.alpha_base = alpha_base
        ctx.kappa = kappa

        return (u >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        u, v_th = ctx.saved_tensors
        D = ctx.d_tracker.get_D()

        alpha = ctx.alpha_base / (1.0 + ctx.kappa * D)

        # Surrogate derivative from Eq. 4 in the paper
        grad_u = (1.0 / (2.0 * alpha)) * (1.0 + (u - v_th).abs() / alpha).pow(-2)

        return grad_output * grad_u, None, None, None, None


class LIFNode(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with:
    - Learnable firing threshold (per-layer)
    - Membrane shortcuts
    - DA-PSG backward pass
    - Detached temporal state to prevent OOM
    """
    def __init__(self, v_th=0.5, tau=4.0, alpha_base=1.0, kappa=10.0, learnable_th=True):
        super().__init__()
        if learnable_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
        self.decay = 1.0 - (1.0 / tau)  # tau=4 -> decay=0.75
        self.alpha_base = alpha_base
        self.kappa = kappa
        
        # Dopamine receptor density and baseline leak
        self.w_d = nn.Parameter(torch.tensor(0.0))
        # Initialized so sigmoid(b_lambda) == decay
        import math
        init_b = math.log(self.decay / (1.0 - self.decay))
        self.b_lambda = nn.Parameter(torch.tensor(init_b))

        self.u = None
        self.epsilon = None
        self.prev_spike = None

    def reset_state(self):
        self.u = None
        self.epsilon = None
        self.prev_spike = None

    def forward(self, x, d_tracker, residual_u=None):
        """
        x: Input current at time t
        d_tracker: DopamineTracker instance
        residual_u: Optional membrane shortcut from a previous block
        Returns: (spike, effective_u)
        """
        if self.u is None:
            self.u = torch.zeros_like(x)
            self.epsilon = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)

        # Wrap in a device tensor so XLA keeps D_t in the computation graph
        # rather than inlining a changing Python scalar (which forces graph retracing).
        D_t = torch.tensor(d_tracker.get_D(), dtype=x.dtype, device=x.device)

        # Dynamic leak controlled by dopamine
        lambda_t = torch.sigmoid(self.w_d * D_t + self.b_lambda)

        # Immediate impact: d(lambda)/d(w_d) = lambda * (1 - lambda) * D_t
        immediate_impact = (self.u * (1.0 - self.prev_spike)) * (lambda_t * (1.0 - lambda_t) * D_t)

        # Update eligibility trace
        self.epsilon = immediate_impact + lambda_t * self.epsilon

        # Hard reset integration
        self.u = lambda_t * self.u * (1.0 - self.prev_spike) + x

        # Membrane shortcut
        if residual_u is not None:
            effective_u = self.u + residual_u
        else:
            effective_u = self.u

        # Spiking with DA-PSG surrogate
        spike = DAPSG.apply(effective_u, self.v_th, d_tracker, self.alpha_base, self.kappa)
        self.prev_spike = spike.detach()

        # E-prop Gradient Accumulation
        # Accumulate the gradient manually so PyTorch optimizer can step without BPTT
        delta_w_d = D_t * self.epsilon.sum()
        
        if self.w_d.grad is None:
            self.w_d.grad = torch.zeros_like(self.w_d)
        
        # PyTorch applies w = w - lr * grad. To do \Delta w_d = \eta * D * \epsilon, we set grad to -delta_w_d.
        self.w_d.grad -= delta_w_d

        # Fully detach temporal states to eliminate BPTT VRAM overhead!
        self.u = self.u.detach()
        self.epsilon = self.epsilon.detach()

        return spike, effective_u
