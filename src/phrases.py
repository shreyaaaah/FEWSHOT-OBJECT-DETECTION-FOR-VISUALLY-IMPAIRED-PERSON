"""Uncertainty-aware spoken phrases (core novelty #5): the wording changes with
the calibrated confidence so the user is never told something falsely certain."""
from typing import Optional

from .schema import Match


def identity_phrase(match: Optional[Match], coco_label: Optional[str] = None) -> str:
    if match is None or match.level == "empty":
        if coco_label:
            return f"I see a {coco_label}. You have not taught me any objects yet."
        return "You have not taught me any objects yet."
    if match.level == "high":
        return f"This is your {match.name}."
    if match.level == "medium":
        return f"This might be your {match.name}."
    if match.level == "ambiguous":
        return (f"This could be your {match.best_name} or your {match.second_name}. "
                f"Please show it from another angle.")
    # unknown
    if coco_label:
        return (f"I see a {coco_label}, but it does not match anything you taught me.")
    return "I am not sure. Try holding it steady and a little closer."


def find_not_found(target: str) -> str:
    return f"I cannot see {target} right now. Turn slowly and I will keep looking."


def hedge_prefix(level: str) -> str:
    """Prefix for spatial answers depending on confidence level."""
    return {"high": "", "medium": "I think ", "low": "I am not certain, but I think "}.get(level, "")


def help_text() -> str:
    return ("You can say: where is my bottle; what is this; what is around me; "
            "save this as keys; forget keys; or what do you know.")
