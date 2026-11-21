"""LSTM trajectory predictor (Pallavi's model).

Idea in one paragraph: read the last 1 s of motion (11 steps of x, y, vx, vy) with an LSTM, then
turn its final summary vector into all 30 future positions at once with a small MLP.
Predicting all 30 points in one shot (instead of one at a time) avoids errors snowballing.

Two options worth knowing about:
  residual=True : the network predicts a CORRECTION to the constant-velocity path instead of the
                  path itself. This starts the model from the baseline, so it can only do better
                  by learning something the baseline misses (braking, turning).
  normalization : inputs are standardised with the training-set mean/std (stored inside the model
                  so a saved checkpoint is self-contained). Outputs are scaled by y_scale meters.
"""
import numpy as np
import torch
from torch import nn

from common import DT, N_FUT


class LSTMPredictor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.1, horizon=N_FUT, residual=False,
                 y_scale=10.0, x_mean=None, x_std=None):
        super().__init__()
        self.in_dim, self.horizon, self.residual, self.y_scale = in_dim, horizon, residual, float(y_scale)
        self.register_buffer("x_mean", torch.zeros(in_dim) if x_mean is None else torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(in_dim) if x_std is None else torch.as_tensor(x_std, dtype=torch.float32))
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, horizon * 2))

    def forward(self, x):
        """x: [B, 11, F] raw inputs in meters (see common.build_inputs). Returns [B, 30, 2] meters."""
        h, _ = self.lstm((x - self.x_mean) / self.x_std)
        out = self.head(h[:, -1]).view(-1, self.horizon, 2) * self.y_scale
        if self.residual:
            out = out + self._cv(x)
        return out

    def _cv(self, x):
        v = (x[:, -1, :2] - x[:, -2, :2]) / DT
        t = torch.arange(1, self.horizon + 1, device=x.device, dtype=x.dtype).view(1, -1, 1) * DT
        return x[:, -1:, :2] + v[:, None, :] * t


def ade_loss(pred, gt):
    """Mean Euclidean distance over all steps (the ADE itself, so we train on what we report)."""
    return torch.sqrt(((pred - gt) ** 2).sum(-1) + 1e-9).mean()


def fit_normalizer(x):
    """Per-feature mean/std of the training inputs. x: [N, 11, F] numpy."""
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), np.maximum(flat.std(0), 1e-3)


def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["cfg"]
    model = LSTMPredictor(cfg["in_dim"], cfg["hidden"], cfg["layers"], cfg["dropout"],
                          residual=cfg["residual"], y_scale=cfg["y_scale"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict(model, x, batch_size=2048, device="cpu"):
    """numpy [N, 11, F] -> numpy [N, 30, 2]."""
    model.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], device=device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)
