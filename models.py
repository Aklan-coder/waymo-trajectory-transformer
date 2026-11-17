import torch
import torch.nn as nn


class TrajTransformer(nn.Module):
    """Transformer encoder over the 11 history steps.
    K=1 -> single trajectory (fair comparison with the LSTM).
    K=3 -> multimodal head: K trajectories + K confidence logits."""

    def __init__(self, in_dim=6, d_model=128, nhead=4, num_layers=3,
                 t_in=11, t_out=30, K=1, dropout=0.1):
        super().__init__()
        self.K, self.t_out = K, t_out
        self.inp = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))          # summary token
        self.pos = nn.Parameter(torch.zeros(1, t_in + 1, d_model))   # learned positions
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.traj_head = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(),
                                       nn.Linear(256, K * t_out * 2))
        self.conf_head = nn.Linear(d_model, K)

    def forward(self, x):                       # x: (B, 11, in_dim)
        B = x.shape[0]
        h = torch.cat([self.cls.expand(B, -1, -1), self.inp(x)], 1) + self.pos
        h = self.norm(self.enc(h)[:, 0])        # CLS output
        traj = self.traj_head(h).view(B, self.K, self.t_out, 2)
        return traj, self.conf_head(h)          # (B,K,30,2), (B,K)


def displacement(pred, gt):
    """pred (B,K,T,2), gt (B,T,2) -> per-step L2 distance (B,K,T)."""
    return torch.linalg.norm(pred - gt[:, None], dim=-1)


def loss_fn(traj, logits, gt, conf_weight=0.5):
    d = displacement(traj, gt)                  # (B,K,T)
    ade = d.mean(-1)                            # (B,K)
    if traj.shape[1] == 1:
        return ade.mean()
    best = ade.argmin(1)                        # winner-takes-all (best-of-K)
    reg = ade.gather(1, best[:, None]).mean()
    cls = nn.functional.cross_entropy(logits, best)
    return reg + conf_weight * cls


def constant_velocity(hist, t_out=30, dt=0.1):
    """Baseline in the agent frame: hist[..., 2:4] is velocity at each step."""
    v = hist[:, -1, 2:4]                                        # (B,2)
    t = torch.arange(1, t_out + 1, device=hist.device) * dt    # (T,)
    return (v[:, None, :] * t[None, :, None])[:, None]          # (B,1,T,2)
