"""Temporal smoothing so the spoken output does not flicker."""
from collections import deque
from typing import Optional, Tuple


class VoteSmoother:
    """Recency-weighted vote over the last `window` frame decisions.
    A decision is a name, or None for 'unknown'."""

    def __init__(self, window=7, min_votes=3, decay=0.85, min_share=0.5):
        self.window = window
        self.min_votes = min_votes
        self.decay = decay
        self.min_share = min_share
        self.buf = deque(maxlen=window)
        self.stable: Optional[str] = None

    def reset(self):
        self.buf.clear()
        self.stable = None

    def update(self, name: Optional[str], score: float = 1.0) -> Tuple[Optional[str], bool]:
        """Returns (stable_name_or_None, changed_since_last_call)."""
        self.buf.append((name, max(score, 1e-3)))
        tally = {}
        counts = {}
        total = 0.0
        for age, (n, s) in enumerate(reversed(self.buf)):
            w = s * (self.decay ** age)
            tally[n] = tally.get(n, 0.0) + w
            counts[n] = counts.get(n, 0) + 1
            total += w
        best = max(tally, key=tally.get)
        if counts[best] >= self.min_votes and tally[best] / total >= self.min_share:
            new = best
        else:
            new = self.stable          # not enough evidence: keep previous decision
        changed = new != self.stable
        self.stable = new
        return new, changed


class LocationFilter:
    """Exponential moving average of an object's box centre / distance.
    Resets if the object disappears for `patience` frames or jumps far."""

    def __init__(self, alpha=0.5, patience=5, jump_frac=0.35):
        self.alpha = alpha
        self.patience = patience
        self.jump_frac = jump_frac
        self.state = None            # (cx, cy, dist)
        self.missed = 0

    def reset(self):
        self.state = None
        self.missed = 0

    def update(self, cx, cy, dist, frame_w):
        self.missed = 0
        if self.state is None or abs(cx - self.state[0]) > self.jump_frac * frame_w:
            self.state = (cx, cy, dist)
        else:
            a = self.alpha
            pcx, pcy, pd = self.state
            d = dist if (dist is None or pd is None) else a * dist + (1 - a) * pd
            self.state = (a * cx + (1 - a) * pcx, a * cy + (1 - a) * pcy, d)
        return self.state

    def miss(self):
        self.missed += 1
        if self.missed >= self.patience:
            self.state = None
        return self.state
