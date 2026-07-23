from torch.amp import custom_fwd, custom_bwd
import torch
import torch.nn as nn
import math


class IntrinsicDopamineTracker:
    """
    Autonomous Dopamine generator using Shannon Entropy to track the
    network's Intrinsic Reward Prediction Error (RPE).

    H[t] = - sum(P_c * log(P_c))
    R[t] = -(H[t] - H[t-1])
    V[t] = beta * V[t-1] + (1 - beta) * R[t]
    D[t] = R[t] - V[t-1]

    Change 3 (consensus): Symmetric clip [-0.3, 0.3].
    Negative RPE (DA dip) enables LTD. TD convergence requires E[D]=0.
    """
    def __init__(self, beta=0.7):
        self.beta = beta
        self.V_t = 0.0
        self.prev_H = None
        self.current_D = 0.0

    def update(self, logits: torch.Tensor):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
            if self.prev_H is None:
                self.prev_H = entropy
                self.current_D = 0.0
                return
            R_t = -(entropy - self.prev_H)
            self.prev_H = entropy
            self.current_D = R_t - self.V_t
            self.V_t = self.beta * self.V_t + (1.0 - self.beta) * R_t

    def get_D(self):
        # Change 3: Symmetric clip [-0.3, 0.3].
        # Negative RPE -> LTD. TD convergence requires E[D]=0 at convergence.
        # Asymmetric [0,0.3] was mathematically broken (always positive bias).
        return max(-0.3, min(self.current_D, 0.3))


class DAPSG(torch.autograd.Function):
    """
    Forward:  s = Theta(u - V_th)
    Backward: sigma'_DA(u) = 1/(2*alpha) * (1 + |u - V_th|/alpha)^(-2)
              where alpha = alpha_base / (1 + kappa * D(t))
    """
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, u, v_th, d_tracker, alpha_base, kappa, D_local=None):
        ctx.save_for_backward(u, v_th)
        ctx.d_tracker = d_tracker
        ctx.alpha_base = alpha_base
        ctx.kappa = kappa
        ctx.D_local = D_local
        return (u >= v_th).float()

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        u, v_th = ctx.saved_tensors
        if ctx.D_local is not None:
            D = ctx.D_local
        else:
            D = ctx.d_tracker.get_D()
            
        # Use raw signed D (NOT abs(D)):
        # D > 0 (positive RPE) -> denominator > 1 -> smaller alpha -> sharper gradient (LTP)
        # D < 0 (negative RPE) -> denominator < 1 -> larger alpha -> wider gradient (LTD)
        # Clamp denominator >= 0.1 to prevent division collapse if kappa*D < -1
        # alpha may be a tensor if D is D_local
        if isinstance(D, torch.Tensor):
            alpha = ctx.alpha_base / torch.clamp(1.0 + ctx.kappa * D, min=0.1)
        else:
            alpha = ctx.alpha_base / max(1.0 + ctx.kappa * D, 0.1)
        grad_u = (1.0 / (2.0 * alpha)) * (1.0 + (u - v_th).abs() / alpha).pow(-2)
        # Change 2: grad for learnable threshold. dS/dv_th = -dS/du
        grad_v_th = -(grad_output * grad_u).sum()
        # Returns 6 items: u, v_th, d_tracker, alpha_base, kappa, D_local
        return grad_output * grad_u, grad_v_th, None, None, None, None


class TernaryDAPSG(torch.autograd.Function):
    """
    Change 5: Dedicated Ternary DA-PSG with piecewise gradient at origin.

    Forward:  spike = Theta(u-v_th) - Theta(-u-v_th)  in {-1, 0, +1}
    Backward: u>=0 -> grad from positive-threshold path only: g(u-v_th)
              u<0  -> grad from negative-threshold path only: g(-u-v_th)

    Eliminates ghost gradients: without this, combined g(u-v_th)+g(-u-v_th)
    at u=0 equals 2*g(v_th) > 0, updating weights for silent neurons.

    Signed Binary Encoding: +1=glutamatergic excitation, -1=GABAergic inhibition.
    This is the Feedforward Excitation/Inhibition (FFI) motif; not a single
    neuron violating Dale's Principle.
    """
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, u, v_th, d_tracker, alpha_base, kappa, D_local=None):
        ctx.save_for_backward(u, v_th)
        ctx.d_tracker = d_tracker
        ctx.alpha_base = alpha_base
        ctx.kappa = kappa
        ctx.D_local = D_local
        pos = (u >= v_th).float()
        neg = (-u >= v_th).float()
        return pos - neg

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        u, v_th = ctx.saved_tensors
        if ctx.D_local is not None:
            D = ctx.D_local
        else:
            D = ctx.d_tracker.get_D()
            
        # Use raw signed D: negative RPE widens gradient (LTD), positive sharpens (LTP)
        if isinstance(D, torch.Tensor):
            alpha = ctx.alpha_base / torch.clamp(1.0 + ctx.kappa * D, min=0.1)
        else:
            alpha = ctx.alpha_base / max(1.0 + ctx.kappa * D, 0.1)
        grad_u_pos = (1.0 / (2.0 * alpha)) * (1.0 + (u - v_th).abs() / alpha).pow(-2)
        grad_u_neg = (1.0 / (2.0 * alpha)) * (1.0 + (-u - v_th).abs() / alpha).pow(-2)
        # Change 5: Piecewise split at u=0.
        # u>=0 path: gradient from positive-threshold path only g(u-v_th).
        # u<0  path: gradient from negative-threshold path only g(-u-v_th).
        # This eliminates gradient superposition at resting potential (u~0).
        # No deadzone needed: piecewise split halves gradient at origin,
        # and a strict mask would permanently kill near-threshold neurons.
        grad_u = torch.where(u >= 0, grad_u_pos, grad_u_neg)
        grad_v_th = -(grad_output * grad_u).sum()
        # Returns 6 items: u, v_th, d_tracker, alpha_base, kappa, D_local
        return grad_output * grad_u, grad_v_th, None, None, None, None


class LIFNode(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with DA-PSG surrogate and e-prop traces.

    Change 4: Soft reset u <- lambda*(u - v_th*spike) + x
              Preserves sub-threshold residual (AHP modelling).
    """
    def __init__(self, v_th=0.5, tau=4.0, alpha_base=2.0, kappa=3.0, learnable_th=True):
        super().__init__()
        if learnable_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
        self.decay = 1.0 - (1.0 / tau)
        self.alpha_base = alpha_base
        self.kappa = kappa
        self.w_d = nn.Parameter(torch.tensor(0.0))
        init_b = math.log(self.decay / (1.0 - self.decay))
        self.b_lambda = nn.Parameter(torch.tensor(init_b))
        self.u = None
        self.epsilon = None
        self.prev_spike = None

    def reset_state(self):
        self.u = None
        self.epsilon = None
        self.prev_spike = None

    def forward(self, x, d_tracker, residual_u=None, D_local=None):
        if self.u is None:
            self.u = torch.zeros_like(x)
            self.epsilon = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)

        # Clamp v_th to prevent negative threshold explosion
        with torch.no_grad():
            self.v_th.clamp_(min=0.01)

        if D_local is not None:
            D_t = D_local
        else:
            D_t = torch.tensor(d_tracker.get_D(), dtype=x.dtype, device=x.device)
            
        if self.training:
            if D_local is not None:
                D_t = (D_t + (torch.rand_like(D_t) - 0.5) * 0.1).clamp(-0.3, 0.3)
            else:
                D_t = (D_t + (torch.rand((), dtype=x.dtype, device=x.device) - 0.5) * 0.1).clamp(-0.3, 0.3)

        lambda_t = torch.sigmoid(self.w_d * D_t + self.b_lambda)

        with torch.no_grad():
            immediate_impact = (self.u * (1.0 - self.prev_spike)) * (lambda_t * (1.0 - lambda_t) * D_t)
            self.epsilon = immediate_impact + lambda_t * self.epsilon

        # Change 4: Soft reset. spike in {0,1}: u - v_th*1 after spike.
        self.u = lambda_t * (self.u - self.v_th * self.prev_spike) + x

        if residual_u is not None:
            effective_u = self.u + residual_u
        else:
            effective_u = self.u

        spike = DAPSG.apply(effective_u, self.v_th, d_tracker, self.alpha_base, self.kappa, D_local)
        self.prev_spike = spike.detach()

        with torch.no_grad():
            delta_w_d = (D_t * self.epsilon).sum()
            if self.w_d.grad is None:
                self.w_d.grad = torch.zeros_like(self.w_d)
            self.w_d.grad -= delta_w_d

        self.u = self.u.detach()
        self.epsilon = self.epsilon.detach()

        return spike, effective_u


class TernaryLIFNode(nn.Module):
    """
    DA-connected Ternary LIF -- Signed Binary Encoding.

    Output spike in {-1, 0, +1}:
      +1 = glutamatergic excitation (FFI excitation path)
      -1 = GABAergic inhibition (FFI inhibition path)
    Not a single neuron violating Dale's Principle -- an abstracted
    paired excitatory/inhibitory population representation.

    Change 4: Soft reset u <- lambda*(u - v_th*spike) + x.
              For spike=-1: u - v_th*(-1) = u + v_th, pushing negative
              membrane back toward 0. Universal formula for both LIF types.
    Change 5: Uses TernaryDAPSG -- piecewise gradient, no ghost gradients.
    """
    def __init__(self, v_th=0.5, tau=4.0, alpha_base=2.0, kappa=3.0, learnable_th=True):
        super().__init__()
        if learnable_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
        self.alpha_base = alpha_base
        self.kappa = kappa
        self.w_d = nn.Parameter(torch.tensor(0.0))
        decay = 1.0 - (1.0 / tau)
        init_b = math.log(decay / (1.0 - decay))
        self.b_lambda = nn.Parameter(torch.tensor(init_b))
        self.u = None
        self.prev_spike = None

    def reset_state(self):
        self.u = None
        self.prev_spike = None

    def forward(self, x, d_tracker, D_local=None):
        if self.u is None:
            self.u = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)

        # Clamp v_th to prevent negative threshold explosion
        with torch.no_grad():
            self.v_th.clamp_(min=0.01)

        if D_local is not None:
            D_t = D_local
        else:
            D_t = torch.tensor(d_tracker.get_D(), dtype=x.dtype, device=x.device)
            
        if self.training:
            if D_local is not None:
                D_t = (D_t + (torch.rand_like(D_t) - 0.5) * 0.1).clamp(-0.3, 0.3)
            else:
                D_t = (D_t + (torch.rand((), dtype=x.dtype, device=x.device) - 0.5) * 0.1).clamp(-0.3, 0.3)

        lambda_t = torch.sigmoid(self.w_d * D_t + self.b_lambda)

        # Change 4: Soft reset -- universal formula.
        # prev_spike in {-1,0,+1}: v_th*(-1) = -v_th -> u + v_th for neg spikes.
        self.u = lambda_t * (self.u - self.v_th * self.prev_spike) + x

        # Change 5: TernaryDAPSG -- piecewise gradient, no ghost gradients
        spike = TernaryDAPSG.apply(self.u, self.v_th, d_tracker, self.alpha_base, self.kappa, D_local)

        self.prev_spike = spike.detach()
        self.u = self.u.detach()

        return spike
