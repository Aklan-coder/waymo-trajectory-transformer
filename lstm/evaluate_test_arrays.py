"""Score constant velocity (and the LSTM, if --ckpt is given) on Abdulahi's arrays.

    # Step 1, no model needed: check your evaluation reproduces his baseline (test 1.414 / 3.849)
    python evaluate_test_arrays.py --test data/test.npz --out results/cv_check

    # Step 2, final numbers (run on the test set ONCE, after all tuning is done)
    python evaluate_test_arrays.py --test data/test.npz --ckpt runs/lstm_a/best.pt --out results/final

--test can be a FOLDER of per-shard .npz files (Abdulahi's data_N/test).
Writes: pred_lstm.npy (N,30,2) for Abdulahi, summary.csv (ADE/FDE with 95% bootstrap CI), paired_differences.csv, per_example_errors.npz
(same row order as the test arrays, so Abdulahi can add his columns), and figures.
"""
import argparse
import os

import numpy as np
import pandas as pd

from array_data import add_data_args, load_from_args
from baselines import constant_velocity
from common import bootstrap_ci, displacement_errors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p, "test")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--expect_cv_ade", type=float, default=1.414)
    p.add_argument("--expect_cv_fde", type=float, default=3.849)
    p.add_argument("--n_examples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    x, y = load_from_args(args, "test")
    n = len(x)
    preds = {"Constant velocity": constant_velocity(x, "dataset")}   # uses the vx, vy columns (headline baseline)
    extra = {"Constant velocity (last 2 positions)": constant_velocity(x, "position"),
             "Constant velocity (last 0.5 s of positions)": constant_velocity(x, "position_avg")}
    if args.ckpt:
        import torch
        from lstm_model import load_checkpoint, predict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_checkpoint(args.ckpt, device)
        preds["LSTM"] = predict(model, x, device=device)

    errors = {m: displacement_errors(p_, y) for m, p_ in preds.items()}
    rows = []
    for m, p_ in {**preds, **extra}.items():
        a, f = displacement_errors(p_, y)
        (am, alo, ahi), (fm, flo, fhi) = bootstrap_ci(a), bootstrap_ci(f)
        rows.append({"model": m, "n": n, "ADE_mean": am, "ADE_lo": alo, "ADE_hi": ahi, "FDE_mean": fm, "FDE_lo": flo, "FDE_hi": fhi})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)

    cv_ade, cv_fde = errors["Constant velocity"][0].mean(), errors["Constant velocity"][1].mean()
    ok = abs(cv_ade - args.expect_cv_ade) < 0.01 and abs(cv_fde - args.expect_cv_fde) < 0.01
    print(f"\nTest cars: {n}")
    for r in rows:
        print(f"  {r['model']:44s} ADE {r['ADE_mean']:6.3f} [{r['ADE_lo']:.3f}, {r['ADE_hi']:.3f}]   FDE {r['FDE_mean']:6.3f} [{r['FDE_lo']:.3f}, {r['FDE_hi']:.3f}]")
    print(f"\nBaseline check vs Abdulahi ({args.expect_cv_ade} / {args.expect_cv_fde}): "
          f"{'MATCH' if ok else 'MISMATCH - check test set, units, column order, or how he computes the baseline'}"
          f"  (yours {cv_ade:.3f} / {cv_fde:.3f})")

    if "LSTM" in errors:
        diffs = []
        for k, i in (("ADE", 0), ("FDE", 1)):
            d = errors["LSTM"][i] - errors["Constant velocity"][i]
            m, lo, hi = bootstrap_ci(d)
            diffs.append({"metric": k, "LSTM minus CV (m)": m, "lo": lo, "hi": hi, "share LSTM better": float((d < 0).mean())})
        pd.DataFrame(diffs).to_csv(os.path.join(args.out, "paired_differences.csv"), index=False)
        print("\nLSTM minus constant velocity (negative = LSTM better):\n" + pd.DataFrame(diffs).round(3).to_string(index=False))

    for m, p_ in preds.items():          # predictions (N,30,2), same row order as the test files
        np.save(os.path.join(args.out, f"pred_{m.lower().replace(' ', '_')}.npy"), p_.astype(np.float32))
    np.savez_compressed(os.path.join(args.out, "per_example_errors.npz"),
                        **{f"{m.lower().replace(' ', '_')}_{k}": e[i] for m, e in errors.items() for k, i in (("ade", 0), ("fde", 1))})

    from plots import plot_metric_bars, plot_overlays
    summary = {m: {"ADE": bootstrap_ci(e[0]), "FDE": bootstrap_ci(e[1])} for m, e in errors.items()}
    plot_metric_bars(summary, os.path.join(args.out, "fig_ade_fde.png"), title="Prediction error on the test set")
    rng = np.random.default_rng(args.seed)
    k = min(args.n_examples, n)
    plot_overlays(x, y, preds, rng.choice(n, k, replace=False), os.path.join(args.out, "fig_overlay_random.png"),
                  "Random test vehicles (not hand-picked)")
    if "LSTM" in errors:
        plot_overlays(x, y, preds, np.argsort(-errors["LSTM"][1])[:k], os.path.join(args.out, "fig_overlay_worst_lstm.png"),
                      "Largest LSTM errors at 3 s")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
