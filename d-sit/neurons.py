import torch
import torch.nn as nn


class DopamineTracker:
    """
    Tracks the global Dopamine signal D(t) representing the Reward Prediction Error.

    D(t) = E[exp(-gamma * (1 - confidence))]

    When classification error is high (confidence ~ 0), D(t) -> exp(-gamma) ~ 0.
    When classification is correct (confidence ~ 1), D(t) -> exp(0) = 1.
    This controls the surrogate width alpha(D) = alpha_base / (1 + kappa * D).
    """
    def __init__(self, gamma=5.0, ema_decay=0.9):
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.current_D = 0.0

    def update(self, R_t, V_hat_t):
        """
        R_t: Ground truth label indices (B,)
        V_hat_t: Model predicted logits (B, num_classes)
        """
        with torch.no_grad():
            probs = torch.softmax(V_hat_t, dim=-1)
            if R_t.dim() == 1:
                confidence = probs[torch.arange(probs.size(0), device=probs.device), R_t]
            else:
                confidence = (probs * R_t).sum(dim=-1)

            rpe = 1.0 - confidence
            D = torch.exp(-self.gamma * rpe).mean().item()

            self.current_D = self.ema_decay * self.current_D + (1.0 - self.ema_decay) * D

    def get_D(self):
        return self.current_D


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
        self.u = None

    def reset_state(self):
        self.u = None

    def forward(self, x, d_tracker, residual_u=None):
        """
        x: Input current at time t
        d_tracker: DopamineTracker instance
        residual_u: Optional membrane shortcut from a previous block
        Returns: (spike, effective_u)
        """
        if self.u is None:
            self.u = torch.zeros_like(x)

        # Leaky integration (decay old state, add new input)
        self.u = self.u.detach() * self.decay + x

        # Membrane shortcut
        if residual_u is not None:
            effective_u = self.u + residual_u
        else:
            effective_u = self.u

        # Spiking with DA-PSG surrogate
        spike = DAPSG.apply(effective_u, self.v_th, d_tracker, self.alpha_base, self.kappa)

        # Soft reset
        self.u = self.u - spike.detach() * self.v_th.detach()

        return spike, effective_u
