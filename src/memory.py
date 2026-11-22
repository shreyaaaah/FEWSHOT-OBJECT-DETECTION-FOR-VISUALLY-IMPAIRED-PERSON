"""Self-calibrating open-set personal memory  (core novelty #2).

What makes it different from a plain "mean prototype + fixed threshold":

1. Multi-prototype bank per object: the mean embedding plus spherical k-means
   centroids of the enrollment frames, so one object seen from several sides is
   not squashed into a single point.
2. Per-object adaptive rejection threshold, calibrated automatically:
     * genuine scores come from *contiguous-fold cross-validation* over the
       enrollment video (held-out segments = unseen viewpoints), not from
       in-sample frames;
     * impostor scores come from the user's OTHER objects and from background
       patches, so the threshold sits between "this object" and "everything
       else this user owns";
     * every new enrollment re-calibrates all thresholds (the memory adapts as
       the collection grows).
3. Confusable-pair warning at enrollment time ("this looks like your X").
4. Calibrated confidence + margin test + safety flag: objects such as medicine
   get a stricter threshold and are never announced with full confidence unless
   the evidence is strong.
"""
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import Match

CRITICAL_WORDS = {"medicine", "medicines", "tablet", "tablets", "pill", "pills",
                  "drug", "syrup", "insulin", "inhaler", "injection", "capsule"}


def l2n(x, axis=-1, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def norm_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def spherical_kmeans(X, k, iters=25, seed=0):
    """k-means on the unit sphere (cosine). X must be L2-normalised."""
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    for _ in range(1, k):                       # k-means++ style seeding
        d = 1.0 - (X @ X[idx].T).max(1)
        d = np.clip(d, 0, None)
        if d.sum() <= 1e-12:
            idx.append(int(rng.integers(n)))
        else:
            idx.append(int(rng.choice(n, p=d / d.sum())))
    C = X[idx].copy()
    for _ in range(iters):
        assign = (X @ C.T).argmax(1)
        newC = C.copy()
        for c in range(k):
            m = assign == c
            if m.any():
                newC[c] = l2n(X[m].mean(0))
        if np.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return C


def build_bank(X, max_clusters=8):
    """Prototype bank = [mean] + k-means centroids."""
    X = l2n(X)
    mean = l2n(X.mean(0))
    k = min(max_clusters, max(1, len(X) // 6))
    if k <= 1:
        return mean[None].astype(np.float32)
    C = spherical_kmeans(X, k)
    return np.vstack([mean[None], C]).astype(np.float32)


def bank_score(bank, X):
    """Max cosine similarity to any prototype. X: (D,) or (N,D) -> (N,)"""
    X = np.atleast_2d(X)
    return (bank @ X.T).max(0)


def auroc(pos, neg):
    """AUROC where higher score = more 'known'. Pure numpy (rank statistic)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_vals = allv[order]
    i = 0
    r = 1
    while i < len(sorted_vals):                 # average ranks for ties
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + r + (j - i)) / 2.0
        r += j - i + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


@dataclass
class ObjectEntry:
    name: str
    X: np.ndarray                    # (n, D) enrollment frame embeddings (temporal order)
    A: np.ndarray                    # (m, D) augmented embeddings
    category: Optional[str] = None   # detector class hint (e.g. "bottle")
    size_m: Optional[float] = None   # longest side in metres (user supplied)
    critical: bool = False           # safety-critical (medicine etc.)
    bank: np.ndarray = None
    threshold: float = 0.6
    sigma: float = 0.03
    separable: bool = True
    sep_margin: Optional[float] = None
    weak: bool = False               # too few frames to calibrate properly
    g_low: float = 0.0               # low percentile of held-out genuine scores


@dataclass
class EnrollReport:
    name: str
    n_frames: int
    n_prototypes: int
    threshold: float
    sigma: float
    separable: bool
    weak: bool
    confusable: List[Tuple[str, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def speech(self) -> str:
        s = f"Saved your {self.name}."
        if self.confusable:
            others = " and ".join(f"your {n}" for n, _ in self.confusable[:2])
            s += f" It looks similar to {others}. Please show it from a few more angles."
        elif self.weak:
            s += " I only saw it from a few angles, so please show it once more."
        return s


class PersonalMemory:
    def __init__(self, max_clusters=8, reject_percentile=5.0, margin=0.04,
                 min_threshold=0.45, max_threshold=0.90, critical_bonus=0.05,
                 relax=0.03, bg_cap=300, cv_folds=10, cv_gap=0, seed=0):
        self.max_clusters = max_clusters
        self.reject_percentile = reject_percentile
        self.margin = margin
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.critical_bonus = critical_bonus
        self.relax = relax                      # extra slack for query-conditioned search
        self.bg_cap = bg_cap
        self.cv_folds = cv_folds        # blocks held out during calibration
        self.cv_gap = cv_gap            # frames dropped on each side of a held-out block
        self.seed = seed
        self.entries: Dict[str, ObjectEntry] = {}
        self.background = np.zeros((0, 0), dtype=np.float32)

    @classmethod
    def from_config(cls, mcfg):
        return cls(max_clusters=mcfg.get("max_prototypes_per_object", 8),
                   reject_percentile=mcfg.get("reject_percentile", 5),
                   margin=mcfg.get("margin", 0.04),
                   cv_folds=mcfg.get("cv_folds", 10),
                   cv_gap=mcfg.get("cv_gap", 0),
                   min_threshold=mcfg.get("min_threshold", 0.45),
                   max_threshold=mcfg.get("max_threshold", 0.90))

    # ------------------------------------------------------------------ info
    def names(self) -> List[str]:
        return sorted(self.entries)

    def __len__(self):
        return len(self.entries)

    # ---------------------------------------------------------------- enroll
    def _add_background(self, B):
        B = l2n(B)
        if self.background.size == 0:
            self.background = B
        else:
            self.background = np.vstack([self.background, B])
        if len(self.background) > self.bg_cap:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(len(self.background), self.bg_cap, replace=False)
            self.background = self.background[np.sort(keep)]

    def enroll(self, name, X, aug=None, category=None, size_m=None,
               critical=None, background=None, replace=False) -> EnrollReport:
        key = norm_name(name)
        X = l2n(X)
        D = X.shape[1]
        aug = l2n(aug) if aug is not None and len(aug) else np.zeros((0, D), np.float32)
        if background is not None and len(background):
            self._add_background(background)
        auto_critical = any(w in CRITICAL_WORDS for w in key.split())
        if key in self.entries and not replace:       # incremental: add more shots
            e = self.entries[key]
            e.X = np.vstack([e.X, X])
            e.A = np.vstack([e.A, aug]) if len(aug) else e.A
            if category:
                e.category = category
            if size_m:
                e.size_m = size_m
            if critical is not None:
                e.critical = bool(critical) or auto_critical
        else:
            self.entries[key] = ObjectEntry(
                name=key, X=X, A=aug, category=category, size_m=size_m,
                critical=bool(critical) or auto_critical)
        self._calibrate_all()
        return self._report(key, X)

    def forget(self, name) -> bool:
        key = norm_name(name)
        if key in self.entries:
            del self.entries[key]
            self._calibrate_all()
            return True
        return False

    # ----------------------------------------------------------- calibration
    def _cv_genuine(self, X):
        """Held-out genuine scores via blocked CV over the (time-ordered) enrollment
        frames: each block of ~1/cv_folds of the video is scored by a bank built
        from the rest.  cv_folds=2..3 approximates leave-a-whole-view-out (very
        pessimistic); large cv_folds approaches random k-fold (optimistic because
        neighbouring video frames are near-duplicates).  cv_gap drops neighbours."""
        n = len(X)
        if n < 8:
            mean = l2n(X.mean(0))
            return (X @ mean) - 0.05, True      # in-sample is optimistic -> slack
        folds = max(2, min(self.cv_folds, n // 2))
        parts = np.array_split(np.arange(n), folds)
        g = []
        for p in parts:
            lo, hi = p.min() - self.cv_gap, p.max() + self.cv_gap
            train = np.array([i for i in range(n) if i < lo or i > hi])
            if len(train) < 3:
                continue
            bank = build_bank(X[train], self.max_clusters)
            g.extend(bank_score(bank, X[p]).tolist())
        return np.array(g), False

    def _calibrate_all(self):
        for key, e in self.entries.items():
            full = np.vstack([e.X, e.A]) if len(e.A) else e.X
            e.bank = build_bank(full, self.max_clusters)
        for key, e in self.entries.items():
            g, weak = self._cv_genuine(e.X)
            e.weak = weak
            t_g = float(np.percentile(g, self.reject_percentile))
            e.g_low = t_g
            e.sigma = float(max(np.std(g), 0.02))
            negs = [o.X for k, o in self.entries.items() if k != key]
            if self.background.size:
                negs.append(self.background)
            if negs:
                neg = np.vstack(negs)
                imp = bank_score(e.bank, neg)
                t_i = float(np.percentile(imp, 99))
                if t_g > t_i:
                    thr = 0.5 * (t_g + t_i)
                    e.separable = True
                else:
                    thr = t_i + 0.01
                    e.separable = False
                e.sep_margin = t_g - t_i
            else:
                thr = t_g
                e.separable = True
                e.sep_margin = None
            if e.critical:
                thr += self.critical_bonus
            e.threshold = float(np.clip(thr, self.min_threshold, self.max_threshold))

    def _report(self, key, Xnew) -> EnrollReport:
        e = self.entries[key]
        conf = []
        for k, o in self.entries.items():
            if k == key:
                continue
            # confusable if the MEDIAN new-object frame scores on the old object at least
            # as high as the old object's own weak held-out frames do (or vice versa).
            # (Median chosen over p90/p95: those flagged 25-30% of clearly different
            # objects in the synthetic stress test, the median ~1%.)
            fwd = bank_score(o.bank, Xnew)
            bwd = bank_score(e.bank, o.X)
            if (np.percentile(fwd, 50) >= o.g_low) or (np.percentile(bwd, 50) >= e.g_low):
                conf.append((k, float(fwd.mean())))
        conf.sort(key=lambda t: -t[1])
        warns = []
        if e.weak:
            warns.append("few frames: threshold is approximate")
        if not e.separable:
            warns.append("object is hard to separate from other enrolled objects")
        return EnrollReport(name=key, n_frames=len(e.X), n_prototypes=len(e.bank),
                            threshold=e.threshold, sigma=e.sigma,
                            separable=e.separable, weak=e.weak,
                            confusable=conf, warnings=warns)

    # ----------------------------------------------------------------- query
    def score_all(self, emb) -> Dict[str, float]:
        emb = l2n(emb)
        return {n: float((e.bank @ emb).max()) for n, e in self.entries.items()}

    def score_against(self, name, emb) -> float:
        e = self.entries[norm_name(name)]
        return float((e.bank @ l2n(emb)).max())

    def query(self, emb) -> Match:
        if not self.entries:
            return Match(None, None, 0.0, 1.0, 0.0, None, 0.0, "empty", 0.0)
        scores = self.score_all(emb)
        order = sorted(scores, key=scores.get, reverse=True)
        top = order[0]
        s1 = scores[top]
        e = self.entries[top]
        second = order[1] if len(order) > 1 else None
        s2 = scores[second] if second else -1.0
        margin = (s1 - s2) if second else 1.0
        z = (s1 - e.threshold) / e.sigma
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        need = self.margin * (2.0 if e.critical else 1.0)

        if s1 < e.threshold:
            level, name = "unknown", None
        elif (second is not None and margin < need and
              s2 >= self.entries[second].threshold - 0.05):
            level, name = "ambiguous", None
        else:
            name = top
            level = "high" if s1 >= e.threshold + 2 * e.sigma else "medium"
            if e.critical and margin < 2 * self.margin:
                level = "medium"
        return Match(name, top, s1, e.threshold, margin, second, s2, level, prob)

    # ----------------------------------------------------------- persistence
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        meta = {"params": {"max_clusters": self.max_clusters,
                           "reject_percentile": self.reject_percentile,
                           "margin": self.margin, "min_threshold": self.min_threshold,
                           "max_threshold": self.max_threshold,
                           "critical_bonus": self.critical_bonus,
                           "relax": self.relax, "bg_cap": self.bg_cap,
                           "cv_folds": self.cv_folds, "cv_gap": self.cv_gap},
                "objects": []}
        arrays = {"bg": self.background}
        for i, (k, e) in enumerate(self.entries.items()):
            meta["objects"].append({"name": k, "category": e.category,
                                    "size_m": e.size_m, "critical": e.critical})
            arrays[f"X{i}"] = e.X
            arrays[f"A{i}"] = e.A
        with open(os.path.join(directory, "memory.json"), "w") as f:
            json.dump(meta, f, indent=2)
        np.savez_compressed(os.path.join(directory, "memory.npz"), **arrays)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "memory.json")) as f:
            meta = json.load(f)
        z = np.load(os.path.join(directory, "memory.npz"))
        mem = cls(**meta["params"])
        mem.background = z["bg"]
        for i, o in enumerate(meta["objects"]):
            mem.entries[o["name"]] = ObjectEntry(
                name=o["name"], X=z[f"X{i}"], A=z[f"A{i}"], category=o["category"],
                size_m=o["size_m"], critical=o["critical"])
        if mem.entries:
            mem._calibrate_all()
        return mem
