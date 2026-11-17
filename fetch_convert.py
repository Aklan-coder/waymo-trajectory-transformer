"""
Download Waymo training files in batches and convert them in parallel (for Chameleon).
Raw files are deleted right after conversion, so disk use stays small.

    python fetch_convert.py --split training --start 0 --end 1000 --out data_full/train --workers 16
    python fetch_convert.py --split validation --start 0 --end 5  --out data_full/val
    python fetch_convert.py --split validation --start 5 --end 10 --out data_full/test

Safe to re-run: files already converted are skipped.
"""
import argparse, os, subprocess, time
from multiprocessing import Pool
import numpy as np
from preprocess_womd import process_shard

BUCKET = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
TOTAL = {"training": 1000, "validation": 150}


def convert(args):
    path, out, idx = args
    d, n_scen = process_shard(path, keep_static=0.1, seed=idx)
    name = os.path.join(out, os.path.basename(path).replace(".tfrecord", "") + ".npz")
    np.savez_compressed(name, **d)
    os.remove(path)
    return os.path.basename(path), n_scen, len(d["hist"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["training", "validation"], required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="raw_tmp")
    ap.add_argument("--batch", type=int, default=32, help="files downloaded per round")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.raw, exist_ok=True)
    tot = TOTAL[a.split]
    todo = [i for i in range(a.start, a.end)
            if not os.path.exists(os.path.join(a.out, f"{a.split}-{i:05d}-of-{tot:05d}.npz"))]
    print(f"{len(todo)} files to fetch ({a.end - a.start - len(todo)} already done), {a.workers} workers", flush=True)
    samples, t0 = 0, time.time()
    for b in range(0, len(todo), a.batch):
        ids = todo[b:b + a.batch]
        urls = [f"{BUCKET}/{a.split}/{a.split}.tfrecord-{i:05d}-of-{tot:05d}" for i in ids]
        subprocess.run(["gsutil", "-m", "-q", "cp", *urls, a.raw + "/"], check=True)
        jobs = [(os.path.join(a.raw, os.path.basename(u)), a.out, i) for u, i in zip(urls, ids)]
        with Pool(min(a.workers, len(jobs))) as pool:
            for name, n_scen, n in pool.imap_unordered(convert, jobs):
                samples += n
        done = b + len(ids)
        rate = (time.time() - t0) / done
        print(f"[{done}/{len(todo)}] files done | samples this run {samples} | "
              f"elapsed {(time.time()-t0)/60:.1f} min | ~{rate*(len(todo)-done)/60:.0f} min left", flush=True)
    print("DONE. samples this run:", samples)
