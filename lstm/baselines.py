"""Constant-velocity baseline: assume the vehicle keeps moving at its latest velocity.

Three ways to get "the latest velocity", all reported so the baseline is not a straw man:
  * "dataset"     : the velocity_x / velocity_y that Waymo stores for the current step (HEADLINE baseline)
  * "position"    : from the last two positions,  v = (p_now - p_previous) / 0.1 s   (noisy)
  * "position_avg": average over the last 0.5 s of positions, v = (p_now - p_5_steps_ago) / 0.5 s

Inputs are the arrays from common.build_inputs (columns: x, y, vx, vy, ...). The prediction is the
same in any rotation of the axes, so it does not matter whether rotate=True or False was used.
"""
import numpy as np

from common import DT, N_FUT


def constant_velocity(x, variant="position"):
    """x: [N, 11, F] model inputs. Returns predicted future x, y: [N, 30, 2] (meters)."""
    pos = x[:, :, :2]
    if variant == "position":
        v = (pos[:, -1] - pos[:, -2]) / DT
    elif variant == "position_avg":
        v = (pos[:, -1] - pos[:, -6]) / (5 * DT)
    elif variant == "dataset":
        v = x[:, -1, 2:4]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    t = (np.arange(1, N_FUT + 1) * DT)[None, :, None]
    return pos[:, -1:, :] + v[:, None, :] * t
