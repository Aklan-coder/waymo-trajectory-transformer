"""Load Abdulahi's preprocessed arrays: inputs [N, 11, 6] and targets [N, 30, 2] (metres).

Accepts either
  * one .npz file holding both arrays (keys are found by SHAPE, so any key names work), or
  * two .npy files (inputs and targets).
Use x_key / y_key to force specific key names if the automatic detection is ambiguous.

Feature order is assumed to be: x, y, vx, vy, then the two heading-change columns.
The first four columns (x, y, vx, vy) are what the constant-velocity baseline uses.
"""
import glob
import os

import numpy as np

N_HIST, N_FUT, N_FEAT = 11, 30, 6


def _pick(arrays, shape_tail, key, what):
    if key is not None:
        return arrays[key]
    hits = [k for k, a in arrays.items() if a.ndim == len(shape_tail) + 1 and tuple(a.shape[1:]) == shape_tail]
    if len(hits) != 1:
        raise ValueError(f"Could not find a unique {what} array of shape (N,{','.join(map(str, shape_tail))}). "
                         f"Candidates: {hits}. All keys: { {k: a.shape for k, a in arrays.items()} }. "
                         f"Pass x_key / y_key to choose.")
    return arrays[hits[0]]


def _load_dir(folder, x_key="hist", y_key="fut"):
    """Abdulahi's layout: a folder of per-shard .npz files (keys hist, fut, ...).
    Files are read in ALPHABETICAL order, the same order his analysis uses."""
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {folder}")
    xs, ys = [], []
    for f in files:
        with np.load(f) as d:
            if d[x_key].size == 0:          # a shard can end up empty
                continue
            xs.append(d[x_key]); ys.append(d[y_key])
    return np.concatenate(xs), np.concatenate(ys)


def load_xy(npz=None, x_path=None, y_path=None, x_key=None, y_key=None, mmap=False):
    if npz and os.path.isdir(npz):
        x, y = _load_dir(npz, x_key or "hist", y_key or "fut")
    elif x_path and y_path:
        x = np.load(x_path, mmap_mode="r" if mmap else None)
        y = np.load(y_path, mmap_mode="r" if mmap else None)
    elif npz:
        with np.load(npz, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        x = _pick(arrays, (N_HIST, N_FEAT), x_key, "input")
        y = _pick(arrays, (N_FUT, 2), y_key, "target")
    else:
        raise ValueError("Give either an .npz file or both x and y .npy files.")
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if x.shape[1:] != (N_HIST, N_FEAT) or y.shape[1:] != (N_FUT, 2) or len(x) != len(y):
        raise ValueError(f"Unexpected shapes: inputs {x.shape} (want N,{N_HIST},{N_FEAT}), targets {y.shape} (want N,{N_FUT},2)")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Found NaN/inf in the arrays. Ask Abdulahi how invalid values are handled.")
    return x, y


def add_data_args(p, name):
    """Add --NAME, --NAME_x, --NAME_y, --NAME_x_key, --NAME_y_key to an argparse parser."""
    p.add_argument(f"--{name}", help=f"{name} .npz file with inputs and targets")
    p.add_argument(f"--{name}_x", help=f"{name} inputs .npy")
    p.add_argument(f"--{name}_y", help=f"{name} targets .npy")
    p.add_argument(f"--{name}_x_key", default=None)
    p.add_argument(f"--{name}_y_key", default=None)


def load_from_args(args, name):
    g = lambda s: getattr(args, f"{name}{s}")
    return load_xy(npz=g(""), x_path=g("_x"), y_path=g("_y"), x_key=g("_x_key"), y_key=g("_y_key"))
