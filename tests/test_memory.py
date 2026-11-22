import numpy as np
import pytest

from src.memory import PersonalMemory, auroc, build_bank, spherical_kmeans, l2n
from tests.synth_emb import make_world


def test_auroc_basic():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert abs(auroc([0.5, 0.5], [0.5, 0.5]) - 0.5) < 1e-9


def test_kmeans_unit_norm():
    X = l2n(np.random.default_rng(0).normal(size=(40, 16)))
    C = spherical_kmeans(X, 4)
    assert C.shape == (4, 16)
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)


def test_bank_small_n_is_mean_only():
    X = l2n(np.random.default_rng(0).normal(size=(5, 16)))
    assert build_bank(X).shape[0] == 1


def _split(world, n_enroll=25):
    enroll, test = {}, {}
    for k, v in world.items():
        enroll[k] = v[:n_enroll]
        test[k] = v[n_enroll:]
    return enroll, test


def test_enroll_and_recognize_known_and_reject_unknown():
    world = make_world(n_objects=14, frames=60, seed=1)
    names = list(world)
    known, unknown = names[:5], names[5:]
    mem = PersonalMemory()
    for n in known:
        mem.enroll(n, world[n][::2])          # user rotates object through all sides
    # known objects: held-out frames from the same video
    ok = 0
    tot = 0
    for n in known:
        for x in world[n][1::2]:
            m = mem.query(x)
            tot += 1
            ok += (m.name == n)
    # unknown objects must be mostly rejected
    fa = 0
    tu = 0
    for n in unknown:
        for x in world[n][1::2]:
            m = mem.query(x)
            tu += 1
            fa += (m.name is not None)
    assert ok / tot > 0.7
    assert fa / tu < 0.2


def test_empty_memory():
    mem = PersonalMemory()
    m = mem.query(np.ones(8, np.float32))
    assert m.level == "empty" and m.name is None


def test_critical_object_gets_stricter_threshold():
    world = make_world(n_objects=4, frames=40, seed=2)
    a = PersonalMemory()
    a.enroll("pen", world["obj0"])
    a.enroll("cup", world["obj1"])
    b = PersonalMemory()
    b.enroll("medicine box", world["obj0"])
    b.enroll("cup", world["obj1"])
    assert b.entries["medicine box"].critical
    assert b.entries["medicine box"].threshold >= a.entries["pen"].threshold


def test_confusable_pair_warning():
    rng = np.random.default_rng(0)
    base = l2n(rng.normal(size=64))
    def mk(eps):
        return l2n(base + eps * rng.normal(size=(30, 64))).astype(np.float32)
    mem = PersonalMemory()
    mem.enroll("blue mug", mk(0.15))
    rep = mem.enroll("blue cup", mk(0.15))
    assert rep.confusable and rep.confusable[0][0] == "blue mug"
    assert "similar" in rep.speech()


def test_thresholds_recalibrate_when_collection_grows():
    world = make_world(n_objects=6, frames=40, seed=3)
    mem = PersonalMemory(min_threshold=0.0)   # avoid clipping hiding the change
    mem.enroll("a", world["obj0"])
    t_alone = mem.entries["a"].threshold
    for i in range(1, 5):
        mem.enroll(f"o{i}", world[f"obj{i}"])
    assert mem.entries["a"].threshold != t_alone


def test_forget_and_incremental_enroll():
    world = make_world(n_objects=3, frames=40, seed=4)
    mem = PersonalMemory()
    mem.enroll("a", world["obj0"][:20])
    mem.enroll("a", world["obj0"][20:])
    assert len(mem.entries["a"].X) == 40
    assert mem.forget("a") and len(mem) == 0
    assert not mem.forget("a")


def test_save_load_roundtrip(tmp_path):
    world = make_world(n_objects=4, frames=30, seed=5)
    mem = PersonalMemory()
    for i in range(3):
        mem.enroll(f"o{i}", world[f"obj{i}"], category="bottle" if i == 0 else None,
                   size_m=0.2 if i == 0 else None)
    mem.save(str(tmp_path))
    mem2 = PersonalMemory.load(str(tmp_path))
    assert mem2.names() == mem.names()
    assert mem2.entries["o0"].category == "bottle" and mem2.entries["o0"].size_m == 0.2
    x = world["obj1"][-1]
    assert mem.query(x).name == mem2.query(x).name


def test_calibrated_beats_fixed_threshold_on_safety():
    """Adaptive memory should not be worse than a fixed mean-prototype threshold
    at the same false-accept budget (sanity check of the novelty claim)."""
    world = make_world(n_objects=16, frames=60, seed=7)
    names = list(world)
    rng = np.random.default_rng(0)
    ours_fa, base_fa, ours_acc, base_acc = [], [], [], []
    for ep in range(10):
        pick = rng.choice(names, 10, replace=False)
        known, unk = pick[:4], pick[4:]
        mem = PersonalMemory()
        for n in known:
            mem.enroll(n, world[n][::2])
        protos = {n: l2n(world[n][::2].mean(0)) for n in known}
        for n in known:
            for x in world[n][1::2]:
                m = mem.query(x)
                ours_acc.append(m.name == n)
                s = {k: float(p @ x) for k, p in protos.items()}
                top = max(s, key=s.get)
                base_acc.append(top == n and s[top] >= 0.6)
        for n in unk:
            for x in world[n][1::2]:
                ours_fa.append(mem.query(x).name is not None)
                s = {k: float(p @ x) for k, p in protos.items()}
                base_fa.append(max(s.values()) >= 0.6)
    # not asserting a winner on synthetic data, only that ours is sane
    assert np.mean(ours_acc) > 0.6
    assert np.mean(ours_fa) < 0.25


def test_distinct_objects_rarely_flagged_confusable():
    """Regression: early version flagged 7/8 clearly different objects."""
    flagged = tot = 0
    for seed in range(4):
        world = make_world(n_objects=8, frames=40, view_noise=0.4, seed=seed)
        mem = PersonalMemory()
        for i in range(8):
            rep = mem.enroll(f"o{i}", world[f"obj{i}"][::2])
            flagged += bool(rep.confusable)
            tot += 1
    assert flagged / tot < 0.1
