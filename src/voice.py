"""Voice command parsing (pure, testable) + optional microphone capture."""
import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# spoken word -> COCO class name used by the detector
SYNONYMS = {
    "water bottle": "bottle", "bottle": "bottle", "flask": "bottle",
    "cup": "cup", "mug": "cup", "tea cup": "cup", "coffee cup": "cup",
    "wine glass": "wine glass",
    "phone": "cell phone", "mobile": "cell phone", "cell phone": "cell phone",
    "smartphone": "cell phone", "mobile phone": "cell phone",
    "laptop": "laptop", "computer": "laptop", "notebook computer": "laptop",
    "remote": "remote", "tv remote": "remote", "remote control": "remote",
    "book": "book", "scissors": "scissors", "mouse": "mouse",
    "keyboard": "keyboard", "chair": "chair", "bag": "backpack",
    "backpack": "backpack", "handbag": "handbag", "purse": "handbag",
    "umbrella": "umbrella", "clock": "clock", "vase": "vase", "bowl": "bowl",
    "toothbrush": "toothbrush", "fork": "fork", "knife": "knife",
    "spoon": "spoon", "banana": "banana", "apple": "apple", "orange": "orange",
    "tv": "tv", "television": "tv", "couch": "couch", "sofa": "couch",
    "bed": "bed", "table": "dining table", "person": "person",
    "suitcase": "suitcase", "plant": "potted plant", "door": None,
}

FIND_RE = re.compile(
    r"^(?:where(?:'s| is| are| was)|find|locate|look for|search for|"
    r"help me find|can you find|where can i find|show me)\s+"
    r"(?:is |are )?(?P<det>my |the |a |an |our )?(?P<t>.+)$")
IDENTIFY_RE = re.compile(
    r"^(?:what(?:'s| is)(?: this| that| it)|what am i holding|identify(?: this| it)?|"
    r"tell me what this is|what object is this|what do i have)$")
SCENE_RE = re.compile(
    r"^(?:what(?:'s| is) (?:around|near|nearby|in front of|there)|what do you see|"
    r"describe(?: the)?(?: scene| surroundings| room| area)?|look around)")
ENROLL_RES = [
    re.compile(r"^(?:save|remember|learn|register|teach|store|add)\s+(?:this|it|that)"
               r"(?:\s+object)?\s+as\s+(?:my |a |the )?(?P<t>.+)$"),
    re.compile(r"^this is (?:my|a|the)\s+(?P<t>.+)$"),
    re.compile(r"^(?:learn|remember|add)\s+(?:my |a |the )?(?P<t>.+)$"),
]
FORGET_RE = re.compile(r"^(?:forget|delete|remove)\s+(?:my |the |a )?(?P<t>.+)$")
LIST_RE = re.compile(
    r"^(?:what (?:things|objects|items) do you know|what do you know|list(?: my)?(?: things| objects| items)?|"
    r"what have i taught you)$")
HELP_RE = re.compile(r"^(?:help|what can you do|commands)$")


@dataclass
class Intent:
    kind: str                    # find | identify | scene | enroll | forget | list | help | unknown
    target: Optional[str] = None
    my: bool = False             # user said "my"
    raw: str = ""


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return " ".join(t.split())


def parse_command(text: str) -> Intent:
    t = normalize(text)
    if not t:
        return Intent("unknown", raw=text)
    if IDENTIFY_RE.match(t):
        return Intent("identify", raw=text)
    if SCENE_RE.match(t):
        return Intent("scene", raw=text)
    if LIST_RE.match(t):
        return Intent("list", raw=text)
    if HELP_RE.match(t):
        return Intent("help", raw=text)
    for rx in ENROLL_RES:
        m = rx.match(t)
        if m:
            return Intent("enroll", target=m.group("t").strip(), raw=text)
    m = FORGET_RE.match(t)
    if m:
        return Intent("forget", target=m.group("t").strip(), raw=text)
    m = FIND_RE.match(t)
    if m:
        det = (m.group("det") or "").strip()
        return Intent("find", target=m.group("t").strip(), my=(det in ("my", "our")), raw=text)
    return Intent("unknown", raw=text)


@dataclass
class Target:
    kind: str                    # personal | coco | unknown
    value: Optional[str]         # memory name or COCO label
    spoken: str                  # what to say back


def _fuzzy(word: str, choices: List[str], cutoff=0.72) -> Optional[str]:
    if word in choices:
        return word
    m = difflib.get_close_matches(word, choices, n=1, cutoff=cutoff)
    return m[0] if m else None


def resolve_target(phrase: str, personal_names: List[str]) -> Target:
    """Map a spoken object phrase to a personal object or a COCO class."""
    p = normalize(phrase)
    p = re.sub(r"^(my|the|a|an|our)\s+", "", p)
    hit = _fuzzy(p, personal_names)
    if hit is None:                                    # try individual tokens
        for tok in p.split():
            hit = _fuzzy(tok, personal_names, cutoff=0.75)
            if hit:
                break
    if hit:
        return Target("personal", hit, f"your {hit}")
    # COCO synonyms, longest phrase first
    for key in sorted(SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", p):
            label = SYNONYMS[key]
            if label:
                return Target("coco", label, f"the {label}")
    hit = _fuzzy(p, [k for k in SYNONYMS if SYNONYMS[k]], cutoff=0.8)
    if hit:
        return Target("coco", SYNONYMS[hit], f"the {SYNONYMS[hit]}")
    return Target("unknown", None, p)


# --------------------------------------------------------- microphone (optional)
def listen_once(timeout: float = 6.0, phrase_limit: float = 6.0) -> Optional[str]:
    """Capture one spoken command. Needs `SpeechRecognition` + `PyAudio`.
    Uses Google's free recogniser (online). Returns None on any failure."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as src:
            r.adjust_for_ambient_noise(src, duration=0.4)
            audio = r.listen(src, timeout=timeout, phrase_time_limit=phrase_limit)
        return r.recognize_google(audio)
    except Exception:
        return None


def get_command(prompt: str = "Command> ", use_mic: bool = True) -> str:
    """Microphone if available, otherwise typed input (keeps the demo usable anywhere)."""
    if use_mic:
        txt = listen_once()
        if txt:
            print(f"[heard] {txt}")
            return txt
    try:
        return input(prompt)
    except EOFError:
        return ""
