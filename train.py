"""
Train / evaluate the Transformer.
    python train.py --data data --K 1 --epochs 30      # single-output Transformer
    python train.py --data data --K 3 --epochs 30      # multimodal extension
    python train.py --data data --eval_cv              # constant-velocity sanity check
"""
import argparse, glob, os, csv, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import TrajTransformer, loss_fn, displacement, constant_velocity


def load(split_dir, max_samples=None):
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    assert files, f"no .npz in {split_dir}"
    H = np.concatenate([np.load(f)["hist"] for f in files])
    F = np.concatenate([np.load(f)["fut"] for f in files])
    if max_samples:
        H, F = H[:max_samples], F[:max_samples]
    return torch.from_numpy(H), torch.from_numpy(F)


@torch.no_grad()
def evaluate(model, loader, device, cv=False):
    if model is not None:
        model.eval()
    rows = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if cv:
            traj, logits = constant_velocity(x), torch.zeros(len(x), 1, device=device)
        else:
            traj, logits = model(x)
        d = displacement(traj, y)                         # (B,K,T)
        ade, fde = d.mean(-1), d[..., -1]                 # (B,K)
        top = logits.argmax(1, keepdim=True)
        ends = traj[:, :, -1]                             # mode spread (collapse check)
        spread = torch.cdist(ends, ends).mean((1, 2)) if traj.shape[1] > 1 else torch.zeros(len(x), device=device)
        rows.append(torch.stack([ade.gather(1, top)[:, 0], fde.gather(1, top)[:, 0],
                                 ade.min(1).values, fde.min(1).values, spread], 1).cpu())
    r = torch.cat(rows).numpy()
    return dict(ADE=r[:, 0].mean(), FDE=r[:, 1].mean(), minADE=r[:, 2].mean(),
                minFDE=r[:, 3].mean(), mode_spread=r[:, 4].mean()), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--max_train", type=int, default=None, help="cap samples (debug on laptop)")
    ap.add_argument("--eval_cv", action="store_true")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    Xv, Yv = load(os.path.join(a.data, "val"))
    val = DataLoader(TensorDataset(Xv, Yv), batch_size=1024)
    if a.eval_cv:
        m, _ = evaluate(None, val, dev, cv=True)
        print("Constant velocity:", {k: round(float(v), 3) for k, v in m.items()})
        return

    Xt, Yt = load(os.path.join(a.data, "train"), a.max_train)
    print(f"train {len(Xt)}  val {len(Xv)}  device {dev}")
    train = DataLoader(TensorDataset(Xt, Yt), batch_size=a.bs, shuffle=True, drop_last=True)

    model = TrajTransformer(d_model=a.d_model, num_layers=a.layers, K=a.K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(train))
    run = os.path.join(a.out, f"transformer_K{a.K}")
    os.makedirs(run, exist_ok=True)
    log = csv.writer(open(os.path.join(run, "log.csv"), "w", newline=""))
    log.writerow(["epoch", "train_loss", "val_ADE", "val_FDE", "val_minADE", "val_minFDE", "mode_spread"])
    best = 1e9
    for ep in range(a.epochs):
        model.train(); tot, t0 = 0.0, time.time()
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(*model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        m, per_ex = evaluate(model, val, dev)
        log.writerow([ep, tot / len(train)] + [float(m[k]) for k in ["ADE", "FDE", "minADE", "minFDE", "mode_spread"]])
        print(f"ep {ep:02d} loss {tot/len(train):.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in m.items()) + f" | {time.time()-t0:.0f}s")
        key = m["minADE"] if a.K > 1 else m["ADE"]
        if key < best:
            best = key
            torch.save(model.state_dict(), os.path.join(run, "best.pt"))
            np.save(os.path.join(run, "val_per_example.npy"), per_ex)  # for paired bootstrap later


if __name__ == "__main__":
    main()
