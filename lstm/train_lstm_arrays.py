"""Train the LSTM on Abdulahi's preprocessed arrays (inputs [N,11,6], targets [N,30,2]).

    python train_lstm_arrays.py --train data/train.npz --val data/val.npz --out runs/lstm_a
    # or separate files:
    python train_lstm_arrays.py --train_x X_train.npy --train_y Y_train.npy \
                                --val_x X_val.npy --val_y Y_val.npy --out runs/lstm_a

The test set is NOT used here. Tune on validation only; run evaluate_test_arrays.py once at the end.
Writes to --out: best.pt, history.csv, loss_curve.png, config.json
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from array_data import add_data_args, load_from_args
from common import displacement_errors
from lstm_model import LSTMPredictor, ade_loss, fit_normalizer, predict, save_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p, "train")
    add_data_args(p, "val")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--residual", action="store_true", help="predict a correction to the constant-velocity path")
    p.add_argument("--max_train", type=int, default=None, help="random subset of N training examples (for CPU / quick runs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xtr, ytr = load_from_args(args, "train")
    xva, yva = load_from_args(args, "val")
    if args.max_train and args.max_train < len(xtr):
        keep = np.sort(rng.choice(len(xtr), args.max_train, replace=False))
        xtr, ytr = xtr[keep], ytr[keep]
    print(f"device={device}  train {len(xtr)}  val {len(xva)}  input {xtr.shape[1:]}  target {ytr.shape[1:]}")

    mean, std = fit_normalizer(xtr)
    model = LSTMPredictor(xtr.shape[-1], args.hidden, args.layers, args.dropout, residual=args.residual,
                          x_mean=mean, x_std=std).to(device)
    cfg = dict(in_dim=int(xtr.shape[-1]), hidden=args.hidden, layers=args.layers, dropout=args.dropout,
               residual=bool(args.residual), y_scale=model.y_scale, rotate=True, use_heading=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**cfg, **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}}, f, indent=2)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    xtr_t, ytr_t = torch.as_tensor(xtr).to(device), torch.as_tensor(ytr).to(device)

    best, bad, rows = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xtr_t), device=device)
        total, count = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = ade_loss(model(xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count

        pred = predict(model, xva, device=device)
        ade, fde = displacement_errors(pred, yva)
        val_ade, val_fde = float(ade.mean()), float(fde.mean())
        sched.step(val_ade)
        rows.append(dict(epoch=epoch, train_ade=train_loss, val_ade=val_ade, val_fde=val_fde,
                         lr=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        flag = ""
        if val_ade < best - 1e-4:
            best, bad, flag = val_ade, 0, "  *best*"
            save_checkpoint(os.path.join(args.out, "best.pt"), model, cfg, extra=dict(epoch=epoch, val_ade=val_ade))
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train ADE {train_loss:6.3f}  val ADE {val_ade:6.3f}  val FDE {val_fde:6.3f}  {rows[-1]['seconds']:.0f}s{flag}", flush=True)
        if bad >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Best val ADE {best:.3f} m. Saved to {args.out}/best.pt")
    try:
        from plots import plot_loss_curves
        plot_loss_curves(os.path.join(args.out, "history.csv"), os.path.join(args.out, "loss_curve.png"))
    except Exception as e:
        print("Could not draw the loss curve:", e)


if __name__ == "__main__":
    main()
