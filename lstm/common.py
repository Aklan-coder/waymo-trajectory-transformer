"""Shared pieces for the Waymo trajectory project.

Pallavi (constant-velocity baseline + LSTM) and Abdullah (Transformer) must agree on
everything in this file, so both models see identical inputs and are scored identically:

  * the time settings (history length, prediction horizon)
  * how raw tracks become model inputs (centering, optional rotation)
  * how errors are measured (ADE / FDE, and minADE@K / minFDE@K for multimodal output)
  * how maneuvers are labelled for the per-maneuver breakdown
"""
import numpy as np

DT = 0.1      # seconds between samples (the Waymo Motion data is recorded at 10 Hz)
N_HIST = 11   # 10 past samples + the current sample (spans 1.0 s)
N_FUT = 30    # 30 future samples = 3.0 s


# --------------------------------------------------------------------------- loading
def load_split(path):
    """Load a preprocessed .npz file into a plain dict of numpy arrays."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------- inputs
def rotate_to_agent_frame(xy, yaw0):
    """Rotate 2-D vectors so the vehicle's heading at the current step points along +x.

    xy   : [N, 2] or [N, T, 2] vectors (positions or velocities) in the world axes
    yaw0 : [N] heading (radians) of each vehicle at the current step
    """
    shape = (-1,) + (1,) * (xy.ndim - 2)
    c = np.cos(yaw0).reshape(shape)
    s = np.sin(yaw0).reshape(shape)
    x = c * xy[..., 0] + s * xy[..., 1]
    y = -s * xy[..., 0] + c * xy[..., 1]
    return np.stack([x, y], axis=-1)


def build_inputs(d, rotate=True, use_heading=False):
    """Turn a loaded split into (inputs, targets) arrays ready for a model.

    inputs  : [N, 11, F] float32. Columns are x, y, vx, vy, then (if use_heading) cos, sin
              of the heading. Positions are meters relative to the vehicle's position at the
              current step, so the last history point is always (0, 0).
    targets : [N, 30, 2] float32, future x, y in meters, same frame as the inputs.

    rotate=True also rotates everything so the vehicle's heading at the current step points
    along +x. Distances (and therefore ADE / FDE) do not change under rotation, but a learned
    model has an easier job when every vehicle faces the same way.
    """
    pos = d["hist_xy"].astype(np.float32)
    vel = d["hist_vel"].astype(np.float32)
    yaw = d["hist_yaw"].astype(np.float32)
    fut = d["fut_xy"].astype(np.float32)
    yaw0 = yaw[:, -1]
    if rotate:
        pos = rotate_to_agent_frame(pos, yaw0)
        vel = rotate_to_agent_frame(vel, yaw0)
        fut = rotate_to_agent_frame(fut, yaw0)
        yaw = yaw - yaw0[:, None]
    feats = [pos, vel]
    if use_heading:
        feats.append(np.stack([np.cos(yaw), np.sin(yaw)], axis=-1))
    x = np.concatenate(feats, axis=-1).astype(np.float32)
    return x, fut.astype(np.float32)


# --------------------------------------------------------------------------- metrics
def displacement_errors(pred, gt):
    """Per-example ADE and FDE in meters (lower is better).

    pred, gt : [N, T, 2].  ADE = mean distance over all T steps, FDE = distance at the last step.
    """
    dist = np.linalg.norm(pred - gt, axis=-1)
    return dist.mean(axis=1), dist[:, -1]


def min_displacement_errors(pred_k, gt):
    """minADE@K and minFDE@K per example, for a model that outputs K candidate paths.

    pred_k : [N, K, T, 2], gt : [N, T, 2].  Each metric takes the best candidate separately.
    Report these separately from single-path ADE / FDE: best-of-K is not a like-for-like score.
    """
    dist = np.linalg.norm(pred_k - gt[:, None], axis=-1)   # [N, K, T]
    return dist.mean(axis=2).min(axis=1), dist[:, :, -1].min(axis=1)


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Mean and (1 - alpha) percentile bootstrap confidence interval.

    Pass per-example errors for a single model, or per-example DIFFERENCES between two models
    (a paired comparison). Returns (mean, low, high).
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- maneuver labels
# These labels use the TRUE future, so they are for grouping results only. Never feed them to a model.
STATIONARY_M = 2.0   # moved less than this far in 3 s  -> "stationary"
TURN_DEG = 20.0      # direction of the last 0.5 s differs from initial heading by more than this -> turn
REVERSE_DEG = 150.0  # ... by more than this -> "other" (reversing / U-turn)


def maneuver_labels(d):
    """Geometric label per example: stationary / straight / left / right / other."""
    yaw0 = d["hist_yaw"][:, -1]
    fut = rotate_to_agent_frame(d["fut_xy"], yaw0)
    final_dist = np.linalg.norm(fut[:, -1], axis=-1)
    last = fut[:, -1] - fut[:, -6]                      # movement over the last 0.5 s
    ang = np.degrees(np.arctan2(last[:, 1], last[:, 0]))
    labels = np.full(len(fut), "straight", dtype=object)
    labels[ang > TURN_DEG] = "left"
    labels[ang < -TURN_DEG] = "right"
    labels[np.abs(ang) > REVERSE_DEG] = "other"
    labels[final_dist < STATIONARY_M] = "stationary"
    return labels.astype(str)


SPEED_DELTA = 2.0    # m/s change between the last observed second and the last predicted second
STOPPED_SPEED = 1.0  # m/s; below this in both seconds the vehicle counts as stopped


def speed_change_labels(d):
    """Speed-profile label per example: stopped / braking / accelerating / steady."""
    n = len(d["fut_xy"])
    v_hist = np.linalg.norm(np.diff(d["hist_xy"], axis=1), axis=-1)[:, -10:].sum(axis=1)   # m in last 1 s = m/s
    path = np.concatenate([np.zeros((n, 1, 2)), d["fut_xy"]], axis=1)                       # starts at the origin
    v_fut = np.linalg.norm(np.diff(path, axis=1), axis=-1)[:, -10:].sum(axis=1)
    dv = v_fut - v_hist
    labels = np.full(n, "steady", dtype=object)
    labels[dv <= -SPEED_DELTA] = "braking"
    labels[dv >= SPEED_DELTA] = "accelerating"
    labels[np.maximum(v_hist, v_fut) < STOPPED_SPEED] = "stopped"
    return labels.astype(str)
