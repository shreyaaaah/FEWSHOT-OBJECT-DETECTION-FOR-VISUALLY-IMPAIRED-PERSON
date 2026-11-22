"""Small shared data classes."""
from dataclasses import dataclass
from typing import Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Det:
    box: Box
    label: str
    conf: float


@dataclass
class Match:
    """Result of querying the personal memory with one embedding."""
    name: Optional[str]          # accepted identity, None if unknown/ambiguous
    best_name: Optional[str]     # closest object even if rejected
    score: float                 # cosine score of best_name
    threshold: float             # adaptive threshold of best_name
    margin: float                # best - second best score
    second_name: Optional[str]
    second_score: float
    level: str                   # high | medium | ambiguous | unknown | empty
    prob: float                  # calibrated confidence in [0, 1]


@dataclass
class Observation:
    box: Box
    label: Optional[str] = None          # COCO / detector label
    match: Optional[Match] = None        # personal identity result
    dist_m: Optional[float] = None
    dist_reliable: bool = True
    source: str = "detector"             # detector | grid | center

    @property
    def name(self):
        return self.match.name if self.match is not None else None
