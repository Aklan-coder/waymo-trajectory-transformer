"""
Figures + statistics for the report.
    !python analyze.py --data data --runs /content/drive/MyDrive/waymo_project/runs
Writes to <runs>/analysis/:
  results_table.csv        overall ADE/FDE/minADE@3/minFDE@3 per model
  by_maneuver.csv          errors split by straight / left / right / braking
  bootstrap_ci.csv         paired bootstrap 95% CIs on per-example error differences
  fig1_single_overlays.png ground truth vs CV vs Transformer(K=1)
  fig2_multimodal.png      3 candidate paths in ambiguous scenes
  fig3_error_bars.png      ADE/FDE bar chart
  fig4_loss_curves.png     training loss + validation ADE per epoch
"""
import argparse, glob, os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import TrajTransformer, constant_velocity

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--runs", default="runs")
ap.add_argument("--d_model", type=int, default=128)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--split", default="val", help="val or test (held-out)")
a = ap.parse_args()
out = os.path.join(a.runs, "analysis" if a.split == "val" else f"analysis_{a.split}"); os.makedirs(out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = "validation" if a.split == "val" else a.split

files = sorted(glob.glob(os.path.join(a.data, a.split, "*.npz")))
H = np.concatenate([np.load(f)["hist"] for f in files]); F = np.concatenate([np.load(f)["fut"] for f in files])
X = torch.from_numpy(H)


@torch.no_grad()
def predict(K):
    m = TrajTransformer(d_model=a.d_model, num_layers=a.layers, K=K).to(dev)
    m.load_state_dict(torch.load(os.path.join(a.runs, f"transformer_K{K}", "best.pt"), map_location=dev))
    m.eval(); T, L = [], []
    for i in range(0, len(X), 2048):
        t, l = m(X[i:i + 2048].to(dev)); T.append(t.cpu()); L.append(l.softmax(-1).cpu())
    return torch.cat(T).numpy(), torch.cat(L).numpy()


cv = constant_velocity(X).numpy()                      # (N,1,30,2)
k1, _ = predict(1)
k3, p3 = predict(3)


def errs(pred):                                        # pred (N,K,30,2) -> per-example ade, fde (N,K)
    d = np.linalg.norm(pred - F[:, None], axis=-1)
    return d.mean(-1), d[..., -1]


ade_cv, fde_cv = [e[:, 0] for e in errs(cv)]
ade_k1, fde_k1 = [e[:, 0] for e in errs(k1)]
A3, F3 = errs(k3); top = p3.argmax(1); idx = np.arange(len(F))
ade_top, fde_top = A3[idx, top], F3[idx, top]
minade, minfde = A3.min(1), F3.min(1)

# ---------- maneuver labels from the ground-truth future (agent frame, x = forward) ----------
# Turn = heading change > 30 deg, measured over the last 0.5 s, only if the car actually moves there
# (otherwise centimetre jitter of a stopped car gives a random direction).
seg = F[:, -1] - F[:, -6]
seg_len = np.linalg.norm(seg, axis=-1)
end_dir = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
v_now = np.linalg.norm(H[:, -1, 2:4], axis=-1)
v_end = seg_len / 0.5
travel = np.linalg.norm(F[:, -1], axis=-1)
man = np.full(len(F), "straight", dtype=object)
man[(v_now > 2) & (v_end < 0.6 * v_now)] = "braking"
moving = seg_len > 1.0                                   # > 2 m/s over the last 0.5 s
man[moving & (end_dir > 30)] = "left turn"
man[moving & (end_dir < -30)] = "right turn"
man[(travel < 3.0) & (v_now < 2)] = "slow/stationary"

# ---------- tables ----------
rows = [("Constant velocity", ade_cv, fde_cv), ("Transformer K=1", ade_k1, fde_k1),
        ("Transformer K=3 (top-confidence)", ade_top, fde_top)]
tab = [{"model": n, "ADE": a_.mean(), "FDE": f_.mean()} for n, a_, f_ in rows]
tab.append({"model": "Transformer K=3 (best of 3)", "minADE@3": minade.mean(), "minFDE@3": minfde.mean()})
pd.DataFrame(tab).round(3).to_csv(os.path.join(out, "results_table.csv"), index=False)

bm = []
for m_ in ["straight", "braking", "left turn", "right turn", "slow/stationary"]:
    s = man == m_
    bm.append({"maneuver": m_, "n": int(s.sum()),
               "CV_ADE": ade_cv[s].mean(), "K1_ADE": ade_k1[s].mean(), "K3_minADE": minade[s].mean(),
               "CV_FDE": fde_cv[s].mean(), "K1_FDE": fde_k1[s].mean(), "K3_minFDE": minfde[s].mean()})
pd.DataFrame(bm).round(3).to_csv(os.path.join(out, "by_maneuver.csv"), index=False)

# ---------- paired bootstrap (resample examples, 2000 times) ----------
rng = np.random.default_rng(0)
def boot(diff, B=2000):
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    return diff.mean(), *np.percentile(bs, [2.5, 97.5])
ci = []
for name, d in [("ADE: CV - K1", ade_cv - ade_k1), ("FDE: CV - K1", fde_cv - fde_k1),
                ("ADE: K1 - K3top", ade_k1 - ade_top), ("FDE: K1 - K3top", fde_k1 - fde_top)]:
    m_, lo, hi = boot(d); ci.append({"comparison": name, "mean_diff_m": m_, "ci_low": lo, "ci_high": hi})
pd.DataFrame(ci).round(3).to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)

# ---------- figures ----------
def pick(label, n):
    c = np.where((man == label) & (v_now > 4))[0]          # clearly moving vehicles only
    return rng.choice(c, min(n, len(c)), replace=False)

ex = np.concatenate([pick("straight", 2), pick("braking", 1), pick("left turn", 2), pick("right turn", 1)])
fig, axs = plt.subplots(2, 3, figsize=(13, 8))
for ax, i in zip(axs.flat, ex):
    ax.plot(H[i, :, 0], H[i, :, 1], "k.-", label="history")
    ax.plot(F[i, :, 0], F[i, :, 1], "g-", lw=2, label="ground truth")
    ax.plot(cv[i, 0, :, 0], cv[i, 0, :, 1], "--", c="gray", label="const. velocity")
    ax.plot(k1[i, 0, :, 0], k1[i, 0, :, 1], "b-", label="Transformer K=1")
    ax.set_title(f"{man[i]}  (ADE {ade_k1[i]:.2f} m)"); ax.axis("equal"); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=8); fig.suptitle(f"Single-trajectory predictions, random moving vehicles ({SPLIT} set, agent frame, metres)")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_single_overlays.png"), dpi=150); plt.close(fig)

ex = np.concatenate([pick("left turn", 2), pick("right turn", 2), pick("braking", 1), pick("straight", 1)])
fig, axs = plt.subplots(2, 3, figsize=(13, 8)); cols = ["tab:red", "tab:blue", "tab:orange"]
for ax, i in zip(axs.flat, ex):
    ax.plot(H[i, :, 0], H[i, :, 1], "k.-"); ax.plot(F[i, :, 0], F[i, :, 1], "g-", lw=3, label="ground truth")
    for k in range(3):
        ax.plot(k3[i, k, :, 0], k3[i, k, :, 1], c=cols[k], label=f"mode {k+1}  p={p3[i, k]:.2f}")
    ax.set_title(f"{man[i]}  minADE@3 {minade[i]:.2f} m"); ax.axis("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.suptitle(f"Multimodal Transformer (K=3): random moving vehicles by maneuver ({SPLIT} set)"); fig.tight_layout()
fig.savefig(os.path.join(out, "fig2_multimodal.png"), dpi=150); plt.close(fig)

names = ["Const. velocity", "Transformer K=1", "K=3 top-conf.", "K=3 best-of-3*"]
fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(4)
ax.bar(x - .2, [ade_cv.mean(), ade_k1.mean(), ade_top.mean(), minade.mean()], .4, label="ADE")
ax.bar(x + .2, [fde_cv.mean(), fde_k1.mean(), fde_top.mean(), minfde.mean()], .4, label="FDE")
ax.set_xticks(x, names); ax.set_ylabel("error (m)"); ax.legend()
ax.set_title(f"{SPLIT.capitalize()} error (*best-of-3 = minADE@3/minFDE@3, not directly comparable)")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_error_bars.png"), dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for K in [1, 3]:
    lg = pd.read_csv(os.path.join(a.runs, f"transformer_K{K}", "log.csv"))
    axs[0].plot(lg.epoch, lg.train_loss, label=f"K={K}")
    axs[1].plot(lg.epoch, lg.val_ADE if K == 1 else lg.val_minADE, label="K=1 val ADE" if K == 1 else "K=3 val minADE@3")
axs[0].set_title("Training loss"); axs[1].set_title("Validation ADE / minADE@3 (m)")
for ax in axs: ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_loss_curves.png"), dpi=150); plt.close(fig)

for f in ["results_table.csv", "by_maneuver.csv", "bootstrap_ci.csv"]:
    print(f"\n== {f} ==\n", pd.read_csv(os.path.join(out, f)).to_string(index=False))
print("\nFigures saved in", out)
