"""Make FAKE arrays in Abdulahi's format, only to test the scripts. Numbers mean nothing for the report.
    python make_fake_arrays.py --out fake --n 20000
"""
import argparse, os
import numpy as np

DT = 0.1

def make(n, rng):
    v0 = rng.uniform(0, 15, n); a = rng.normal(0, 1.0, n); w = rng.normal(0, 0.08, n)
    t = (np.arange(41) - 10) * DT                       # time relative to "now" (index 10)
    v = np.clip(v0[:, None] + a[:, None] * t, 0, None)
    psi = w[:, None] * t                                 # heading change relative to now
    vx, vy = v * np.cos(psi), v * np.sin(psi)
    px = np.cumsum(vx * DT, axis=1); py = np.cumsum(vy * DT, axis=1)
    px -= px[:, 10:11]; py -= py[:, 10:11]               # centre at current position
    xs = np.stack([px, py, vx, vy, np.sin(psi), np.cos(psi)], -1)[:, :11].astype(np.float32)
    ys = np.stack([px, py], -1)[:, 11:].astype(np.float32)
    return xs, ys

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=20000)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
    for name, n in (("train", a.n), ("val", a.n // 10), ("test", a.n // 10)):
        x, y = make(n, rng); np.savez_compressed(f"{a.out}/{name}.npz", X=x, Y=y); print(name, x.shape, y.shape)
