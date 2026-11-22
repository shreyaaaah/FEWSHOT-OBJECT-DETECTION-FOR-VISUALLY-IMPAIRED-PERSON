"""Synthetic embedding world used by tests and by the sanity benchmark."""
import numpy as np


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def make_world(n_objects=12, dim=64, frames=40, view_noise=0.6, n_views=3, seed=0):
    """Each object = a few 'view' directions around a centre, frames follow a
    slow random walk between views (like a rotating video)."""
    rng = np.random.default_rng(seed)
    world = {}
    for o in range(n_objects):
        centre = l2n(rng.normal(size=dim))
        views = l2n(centre + 0.5 * rng.normal(size=(n_views, dim)))
        X = []
        for t in range(frames):
            v = views[int(t * n_views / frames)]
            X.append(l2n(v + 0.25 * view_noise * rng.normal(size=dim)))
        world[f"obj{o}"] = np.array(X, dtype=np.float32)
    return world
