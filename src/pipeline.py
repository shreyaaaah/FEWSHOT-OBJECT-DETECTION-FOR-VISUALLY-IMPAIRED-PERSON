"""The assistant brain: enrollment, identification, query-driven search, scene
description and spoken answers.

Novelty #3  - query-conditioned, detector-free localisation:
    "Where are my keys?" works even though 'keys' is not a detector class: if the
    detector does not produce an acceptable candidate, the frame is searched with
    multi-scale windows scored directly against the stored prototypes of that
    object (the few-shot memory itself acts as the detector).
Novelty #4  - known-size-anchored depth (see depth.py).
"""
import collections
from typing import Dict, List, Optional

import numpy as np

from . import depth as depthlib
from . import phrases, spatial
from .augment import augment_crop, random_background_crops
from .frames import select_frames
from .imgutil import center_box, crop_box, grid_boxes, nms
from .memory import PersonalMemory, norm_name
from .schema import Det, Match, Observation
from .voice import Intent, Target, parse_command, resolve_target

IGNORE_FOR_ENROLL = {"person"}          # hands / people are not the object being taught


class Sahayak:
    def __init__(self, embedder, memory: PersonalMemory, detector=None,
                 depth_model=None, speaker=None, cfg: Optional[dict] = None):
        self.emb = embedder
        self.mem = memory
        self.det = detector
        self.depth = depth_model
        self.speaker = speaker
        self.cfg = cfg or {}
        dcfg = self.cfg.get("depth", {})
        self.hfov = dcfg.get("hfov_deg", 65.0)
        ecfg = self.cfg.get("enroll", {})
        self.n_sample = ecfg.get("frames_to_sample", 40)
        self.keep = ecfg.get("keep_top_k", 25)
        self.min_sharp = ecfg.get("min_sharpness", 40.0)
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------- utilities
    def _detect(self, frame) -> List[Det]:
        return self.det.detect(frame) if self.det is not None else []

    def _pick_central(self, dets: List[Det], W, H, exclude=()) -> Optional[Det]:
        """Large + central detection = the object the user is holding up."""
        best, best_s = None, -1.0
        for d in dets:
            if d.label in exclude:
                continue
            x1, y1, x2, y2 = d.box
            area = (x2 - x1) * (y2 - y1) / float(W * H)
            cx, cy = spatial.box_center(d.box)
            dist = np.hypot((cx - W / 2) / W, (cy - H / 2) / H)
            s = area * (1.0 - min(1.0, dist * 1.5)) * (0.5 + d.conf)
            if s > best_s:
                best, best_s = d, s
        return best

    def _locate_held_object(self, frame):
        H, W = frame.shape[:2]
        d = self._pick_central(self._detect(frame), W, H, exclude=IGNORE_FOR_ENROLL)
        if d is not None:
            return d.box, d.label, "detector"
        return center_box(W, H), None, "center"

    def _real_size(self, o: Observation) -> Optional[float]:
        if o.match is not None and o.match.name:
            e = self.mem.entries.get(o.match.name)
            if e is not None:
                if e.size_m:
                    return e.size_m
                if e.category and e.category in depthlib.SIZE_TABLE:
                    return depthlib.SIZE_TABLE[e.category]
        if o.label and o.label in depthlib.SIZE_TABLE:
            return depthlib.SIZE_TABLE[o.label]
        return None

    def attach_distance(self, frame, obs: List[Observation]) -> List[Observation]:
        if not obs:
            return obs
        H, W = frame.shape[:2]
        f = spatial.focal_px(W, self.hfov)
        size_d, rel = [], []
        for o in obs:
            real = self._real_size(o)
            if real:
                d, ok = depthlib.size_distance(o.box, real, f, W, H)
            else:
                d, ok = None, False
            size_d.append(d)
            rel.append(ok)
        dists = size_d
        used_depth = False
        if self.depth is not None:
            dm = self.depth(frame)
            rv = [depthlib.box_depth_value(dm, o.box) for o in obs]
            dists, info = depthlib.fuse_distances(size_d, rel, rv)
            used_depth = info["used_depth"]
        for i, o in enumerate(obs):
            o.dist_m = dists[i]
            if size_d[i] is not None:
                o.dist_reliable = rel[i]
            else:                                    # distance came from depth cue only
                o.dist_reliable = False if dists[i] is not None else True
        return obs

    # ------------------------------------------------------------ enrollment
    def enroll_from_frames(self, name: str, frames: List[np.ndarray],
                           size_cm: Optional[float] = None, critical=None):
        """Teach a new object from a list of BGR frames (e.g. a 5-10 s video)."""
        idx, _ = select_frames(frames, self.keep, self.min_sharp)
        if len(idx) < 3:
            raise ValueError("not enough usable frames - please record again")
        crops, augs, bgs, labels = [], [], [], []
        for i in idx:
            f = frames[i]
            box, label, _ = self._locate_held_object(f)
            c = crop_box(f, box)
            crops.append(c)
            augs.append(augment_crop(c, self.rng))
            augs.append(augment_crop(c, self.rng))
            bgs.extend(random_background_crops(f, box, n=2, rng=self.rng))
            if label:
                labels.append(label)
        X = self.emb.embed_bgr(crops)
        A = self.emb.embed_bgr(augs)
        B = self.emb.embed_bgr(bgs) if bgs else None
        category = collections.Counter(labels).most_common(1)[0][0] if labels else None
        rep = self.mem.enroll(name, X, aug=A, category=category,
                              size_m=(size_cm / 100.0 if size_cm else None),
                              critical=critical, background=B)
        return rep

    # -------------------------------------------------------- identification
    def identify_held(self, frame) -> Observation:
        """'What is this?' - the object held in front of the camera."""
        H, W = frame.shape[:2]
        d = self._pick_central(self._detect(frame), W, H, exclude=IGNORE_FOR_ENROLL)
        if d is not None:
            box, label, src = d.box, d.label, "detector"
        else:
            box, label, src = center_box(W, H), None, "center"
        e = self.emb.embed_bgr([crop_box(frame, box)])[0]
        o = Observation(box=box, label=label, match=self.mem.query(e), source=src)
        return self.attach_distance(frame, [o])[0]

    def scene(self, frame) -> List[Observation]:
        """Everything the detector sees, each also checked against personal memory."""
        dets = self._detect(frame)
        if not dets:
            return []
        embs = self.emb.embed_bgr([crop_box(frame, d.box) for d in dets])
        obs = [Observation(box=d.box, label=d.label, match=self.mem.query(e))
               for d, e in zip(dets, embs)]
        return self.attach_distance(frame, obs)

    # ---------------------------------------------------------------- search
    def _accept_for_target(self, m: Match, name: str) -> Optional[str]:
        """Query-conditioned acceptance. Returns confidence level or None."""
        if m.best_name != name:
            return None
        e = self.mem.entries[name]
        if m.score >= e.threshold + 2 * e.sigma and m.level in ("high", "medium"):
            return "high" if m.level == "high" else "medium"
        if m.score >= e.threshold and m.level != "ambiguous":
            return "medium"
        if m.score >= e.threshold - self.mem.relax:
            return "low"
        return None

    def find_personal(self, frame, name: str) -> List[Observation]:
        name = norm_name(name)
        e = self.mem.entries[name]
        H, W = frame.shape[:2]
        cands: List[Observation] = []
        # 1) detector candidates (restricted to the object's category when known)
        dets = self._detect(frame)
        if e.category:
            same = [d for d in dets if d.label == e.category]
            dets = same if same else dets
        sources = [(d.box, d.label, "detector") for d in dets]
        found = self._score_candidates(frame, sources, name)
        # 2) detector-free fallback: search the frame with multi-scale windows
        if not found:
            sources = [(b, None, "grid") for b in grid_boxes(W, H)]
            found = self._score_candidates(frame, sources, name)
            if found:
                keep = nms([o.box for o in found], [o.match.score for o in found], 0.3)
                found = [found[i] for i in keep]
        found.sort(key=lambda o: -o.match.score)
        return self.attach_distance(frame, found)

    def _score_candidates(self, frame, sources, name) -> List[Observation]:
        if not sources:
            return []
        embs = self.emb.embed_bgr([crop_box(frame, b, pad=0.0) for b, _, _ in sources])
        out = []
        for (box, label, src), e in zip(sources, embs):
            m = self.mem.query(e)
            lvl = self._accept_for_target(m, name)
            if lvl is None:
                continue
            m.level = lvl
            out.append(Observation(box=box, label=label, match=m, source=src))
        return out

    def find_coco(self, frame, label: str) -> List[Observation]:
        dets = [d for d in self._detect(frame) if d.label == label]
        obs = [Observation(box=d.box, label=d.label) for d in dets]
        obs = self.attach_distance(frame, obs)
        obs.sort(key=lambda o: (o.dist_m is None, o.dist_m or 0))
        return obs

    # ------------------------------------------------------------- answering
    def answer_find(self, frame, tgt: Target) -> str:
        H, W = frame.shape[:2]
        if tgt.kind == "personal":
            obs = self.find_personal(frame, tgt.value)
            if not obs:
                return phrases.find_not_found(tgt.spoken)
            o = obs[0]
            pre = phrases.hedge_prefix(o.match.level)
            subject = f"{pre}{tgt.spoken}".strip()
            subject = subject[0].upper() + subject[1:] if subject else subject
            s = spatial.describe_location(subject, o.box, W, H, o.dist_m,
                                          o.dist_reliable, self.hfov)
            if o.match.level == "low":
                s = s.replace(".", ". Please check by touch.", 1) if s.endswith(".") else s
            return s
        if tgt.kind == "coco":
            obs = self.find_coco(frame, tgt.value)
            if not obs:
                return phrases.find_not_found(tgt.spoken)
            o = obs[0]
            subject = tgt.spoken[0].upper() + tgt.spoken[1:]
            s = spatial.describe_location(subject, o.box, W, H, o.dist_m,
                                          o.dist_reliable, self.hfov)
            if len(obs) > 1:
                s += f" I see {len(obs)} in total; this is the nearest."
            return s
        return f"I do not know what {tgt.spoken} is. You can teach me by saying: save this as {tgt.spoken}."

    def answer_identify(self, frame) -> str:
        o = self.identify_held(frame)
        s = phrases.identity_phrase(o.match, o.label)
        H, W = frame.shape[:2]
        if o.match is not None and o.match.level in ("unknown", "ambiguous") and o.source == "center":
            hint = spatial.framing_hint(o.box, W, H)
            if hint:
                s += " " + hint
        return s

    def answer_scene(self, frame) -> str:
        H, W = frame.shape[:2]
        items = []
        for o in self.scene(frame):
            nm = f"your {o.name}" if o.name else o.label
            if nm:
                items.append((nm, o.box, o.dist_m, o.dist_reliable))
        return spatial.scene_phrase(items, W, H, self.hfov)

    def respond(self, frame, text: str) -> str:
        """Handle one spoken command (everything except enrollment, which needs capture)."""
        it: Intent = parse_command(text)
        if it.kind == "identify":
            return self.answer_identify(frame)
        if it.kind == "scene":
            return self.answer_scene(frame)
        if it.kind == "find":
            return self.answer_find(frame, resolve_target(it.target, self.mem.names()))
        if it.kind == "list":
            n = self.mem.names()
            return ("I know your " + ", ".join(n) + ".") if n else "You have not taught me anything yet."
        if it.kind == "forget":
            t = resolve_target(it.target, self.mem.names())
            if t.kind == "personal" and self.mem.forget(t.value):
                return f"I have forgotten your {t.value}."
            return f"I do not know {it.target}."
        if it.kind == "help":
            return phrases.help_text()
        if it.kind == "enroll":
            return "Hold the object and rotate it slowly. I will record now."
        return "Sorry, I did not understand. " + phrases.help_text()
