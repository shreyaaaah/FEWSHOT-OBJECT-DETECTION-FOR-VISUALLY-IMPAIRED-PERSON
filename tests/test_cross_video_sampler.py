import numpy as np

from src.train.datasets import EpisodeSampler


def fake(n_clean=12, n_clutter=15):
    return ([f"/d/clean_v{i // 4}_{i:04d}.jpg" for i in range(n_clean)]
            + [f"/d/clutter_v{i // 5}_{i:04d}.jpg" for i in range(n_clutter)])


def data(n=8):
    return {f"u{j}__obj{j}": fake() for j in range(n)}


def test_cross_video_support_is_clean_and_queries_are_clutter():
    s = EpisodeSampler(data(), list(data()), n_way=4, k_shot=3, q_query=6, cross_video=True, seed=1)
    for _ in range(20):
        ep = s.sample()
        sp, qp, up = ep.support_paths, ep.query_paths, ep.unknown_paths
        assert all("clean_" in p.split("/")[-1] for p in sp)
        assert all("clutter_" in p.split("/")[-1] for p in qp)
        assert all("clutter_" in p.split("/")[-1] for p in up)


def test_default_sampler_unchanged_and_falls_back_without_kinds():
    d = {f"u{j}__o": [f"/d/{i:04d}.jpg" for i in range(20)] for j in range(8)}
    s = EpisodeSampler(d, list(d), n_way=4, k_shot=3, q_query=6, cross_video=True, seed=1)
    ep = s.sample()                       # no clean/clutter names -> falls back to random frames
    assert len(ep.support_paths) == 12 and len(ep.query_paths) == 24
