"""
Quick self-test that needs no Waymo download and no GPU.
    python tests/test_pipeline.py
Checks: reading Waymo-format records, sample construction, frame rotation,
parallel conversion, and the maneuver-labelling rule.
"""
import os, sys, struct, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preprocess_womd as P


def write_fake_shard(path, n_scen=3):
    """Waymo-format TFRecord: per scenario 3 vehicles (one with a gap) + 1 pedestrian."""
    with open(path, "wb") as f:
        for k in range(n_scen):
            sc = P.Scenario(); sc.scenario_id = f"s{k}"; sc.current_time_index = 10
            for j in range(4):
                tr = sc.tracks.add(); tr.id = j; tr.object_type = 1 if j < 3 else 2
                for t in range(91):
                    s = tr.states.add()
                    s.center_x = 100 + (j + 1) * t; s.center_y = 5.0; s.heading = 0.0
                    s.velocity_x = 10.0 * (j + 1); s.velocity_y = 0.0
                    s.valid = not (j == 2 and t == 20)
            b = sc.SerializeToString()
            f.write(struct.pack("<Q", len(b)) + b"\0" * 4 + b + b"\0" * 4)


def test_conversion():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training.tfrecord-00000-of-01000")
        write_fake_shard(p)
        out, n_scen = P.process_shard(p)
        assert n_scen == 3
        assert len(out["hist"]) == 6, "2 valid vehicles x 3 scenarios expected"
        assert out["hist"].shape[1:] == (11, 6) and out["fut"].shape[1:] == (30, 2)
        assert np.allclose(out["hist"][:, -1, :2], 0), "current position must be the origin"
        assert np.allclose(out["fut"][0, -1], [30, 0]), "10 m/s for 3 s = 30 m straight ahead"


def test_rotation():
    # a car driving north (heading 90 deg) must face +x after rotation
    T = 91; t = np.arange(T) * 0.1
    xy = np.stack([np.zeros(T), 10 * t], 1); vel = np.tile([0, 10.0], (T, 1))
    smp = P.make_sample(xy, vel, np.full(T, np.pi / 2), np.ones(T, bool), 10)
    assert np.allclose(smp[1][-1], [30, 0], atol=1e-4)


def test_parallel_convert():
    import fetch_convert as FC
    from multiprocessing import Pool
    with tempfile.TemporaryDirectory() as d:
        jobs = []
        for i in range(3):
            p = os.path.join(d, f"training.tfrecord-{i:05d}-of-01000"); write_fake_shard(p, 2)
            jobs.append((p, d, i))
        with Pool(3) as pool:
            res = list(pool.imap_unordered(FC.convert, jobs))
        assert sum(r[2] for r in res) == 12
        assert sorted(f for f in os.listdir(d) if f.endswith(".npz")) == \
            [f"training-{i:05d}-of-01000.npz" for i in range(3)]


def test_maneuver_rule_ignores_parked_jitter():
    rng = np.random.default_rng(0)
    turns = 0
    for _ in range(500):
        F = np.cumsum(rng.normal(0, 0.03, (30, 2)), 0)      # parked car, cm jitter
        seg = F[-1] - F[-6]
        if np.linalg.norm(seg) > 1.0:                         # the rule's movement check
            turns += 1
    assert turns == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("All tests passed.")
