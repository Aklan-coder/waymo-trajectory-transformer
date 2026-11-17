"""
Convert Waymo Open Motion Dataset (v1.3.1, scenario format) TFRecord files into
small .npz files of vehicle samples: 11 history steps -> 30 future steps.

No TensorFlow or waymo-open-dataset package needed: it reads the TFRecord
framing directly and decodes only the fields we use (other fields are skipped).

Usage (Colab):
    !python preprocess_womd.py --shards "raw/training*"   --out data/train
    !python preprocess_womd.py --shards "raw/validation*" --out data/val
"""
import argparse, glob, os, struct, time
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

T_HIST, T_FUT = 11, 30          # 1.1 s history (incl. current), 3.0 s future at 10 Hz
VEHICLE = 1                     # Track.ObjectType.TYPE_VEHICLE


# ---------- minimal Scenario proto (same field numbers as Waymo's scenario.proto) ----------
def _build_scenario_class():
    T = descriptor_pb2.FieldDescriptorProto
    fdp = descriptor_pb2.FileDescriptorProto(name="mini_scenario.proto", package="mini", syntax="proto2")

    def msg(name, fields):
        m = fdp.message_type.add(name=name)
        for fname, num, ftype, label, tname in fields:
            f = m.field.add(name=fname, number=num, type=ftype, label=label)
            if tname:
                f.type_name = tname

    OPT, REP = T.LABEL_OPTIONAL, T.LABEL_REPEATED
    msg("ObjectState", [("center_x", 2, T.TYPE_DOUBLE, OPT, None),
                        ("center_y", 3, T.TYPE_DOUBLE, OPT, None),
                        ("heading", 8, T.TYPE_FLOAT, OPT, None),
                        ("velocity_x", 9, T.TYPE_FLOAT, OPT, None),
                        ("velocity_y", 10, T.TYPE_FLOAT, OPT, None),
                        ("valid", 11, T.TYPE_BOOL, OPT, None)])
    msg("Track", [("id", 1, T.TYPE_INT32, OPT, None),
                  ("object_type", 2, T.TYPE_INT32, OPT, None),
                  ("states", 3, T.TYPE_MESSAGE, REP, ".mini.ObjectState")])
    msg("Scenario", [("tracks", 2, T.TYPE_MESSAGE, REP, ".mini.Track"),
                     ("scenario_id", 5, T.TYPE_STRING, OPT, None),
                     ("current_time_index", 10, T.TYPE_INT32, OPT, None)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    desc = pool.FindMessageTypeByName("mini.Scenario")
    try:
        return message_factory.GetMessageClass(desc)
    except AttributeError:                       # older protobuf
        return message_factory.MessageFactory(pool).GetPrototype(desc)


Scenario = _build_scenario_class()


def read_tfrecord(path):
    """Yield raw records from a TFRecord file (uncompressed)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)                    # uint64 length + uint32 crc
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            f.read(4)                            # data crc (not checked)
            yield data


# ---------- sample construction ----------
def rotate(v, ang):
    """Rotate (...,2) vectors by -ang (into the agent frame)."""
    c, s = np.cos(ang), np.sin(ang)
    x, y = v[..., 0], v[..., 1]
    return np.stack([c * x + s * y, -s * x + c * y], -1)


def make_sample(xy, vel, heading, valid, cur):
    """xy, vel: (T,2); heading, valid: (T,). cur = current_time_index (10).
    Returns (hist_feat (11,6), fut_xy (30,2), origin (2,), yaw) or None."""
    h0, f1 = cur - T_HIST + 1, cur + T_FUT + 1
    if h0 < 0 or f1 > len(valid) or not valid[h0:f1].all():
        return None
    origin, yaw = xy[cur].copy(), heading[cur]
    hxy = rotate(xy[h0:cur + 1] - origin, yaw)
    hv = rotate(vel[h0:cur + 1], yaw)
    dh = heading[h0:cur + 1] - yaw
    feat = np.concatenate([hxy, hv, np.sin(dh)[:, None], np.cos(dh)[:, None]], -1)
    fut = rotate(xy[cur + 1:f1] - origin, yaw)
    return feat.astype(np.float32), fut.astype(np.float32), origin.astype(np.float32), np.float32(yaw)


def process_shard(path, keep_static=0.1, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["hist", "fut", "origin", "yaw", "scenario"]}
    n_scen = 0
    for rec in read_tfrecord(path):
        sc = Scenario()
        sc.ParseFromString(rec)
        n_scen += 1
        cur = sc.current_time_index
        for tr in sc.tracks:
            if tr.object_type != VEHICLE:
                continue
            st = tr.states
            xy = np.array([[s.center_x, s.center_y] for s in st])
            vel = np.array([[s.velocity_x, s.velocity_y] for s in st])
            hd = np.array([s.heading for s in st])
            va = np.array([s.valid for s in st], bool)
            smp = make_sample(xy, vel, hd, va, cur)
            if smp is None:
                continue
            # parked cars dominate the data; keep only a fraction of them
            if np.linalg.norm(smp[1][-1]) < 2.0 and rng.random() > keep_static:
                continue
            for k, v in zip(["hist", "fut", "origin", "yaw"], smp):
                out[k].append(v)
            out["scenario"].append(sc.scenario_id)
    d = {k: np.array(v) for k, v in out.items()}
    return d, n_scen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_static", type=float, default=0.1)
    ap.add_argument("--delete_raw", action="store_true", help="delete each raw file after processing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shards = sorted(p for s in a.shards for p in glob.glob(s))
    assert shards, f"no files match {a.shards}"
    total = 0
    for i, p in enumerate(shards):
        t0 = time.time()
        d, n_scen = process_shard(p, a.keep_static, seed=i)
        name = os.path.join(a.out, os.path.basename(p).replace(".tfrecord", "") + ".npz")
        np.savez_compressed(name, **d)
        total += len(d["hist"])
        print(f"{os.path.basename(p)}: {n_scen} scenarios -> {len(d['hist'])} samples "
              f"({time.time()-t0:.0f}s)  | total so far {total}", flush=True)
        if a.delete_raw:
            os.remove(p)
    print("DONE. total samples:", total)
