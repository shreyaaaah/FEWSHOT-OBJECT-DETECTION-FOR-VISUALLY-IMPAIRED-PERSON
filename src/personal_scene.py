"""Re-name detector boxes that match something the user taught (the "two phones" case):
the detector says 'cell phone' for both, the personal memory says one of them is 'mobile'."""
from .imgutil import center_box, crop_box


def box_iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class PersonalTagger:
    """Every `every` frames the `max_boxes` most confident non-person boxes (and the middle of the picture,
    for a held object the detector did not box) are embedded and looked up in the personal memory.
    In between, the last answers are reused for boxes that overlap (IoU > 0.4)."""

    def __init__(self, emb, mem, every=3, max_boxes=3):
        self.emb, self.mem = emb, mem
        self.every, self.max_boxes = every, max_boxes
        self.n = 0
        self._cache = []            # [(box, extra or None)]
        self.centre_hit = None      # (box, extra) when a taught object is held in the middle of the picture

    def _extra(self, m):
        if m.name and m.level in ("high", "medium"):
            e = self.mem.entries[m.name]
            return dict(name=m.name, prob=float(m.prob), level=m.level, size_m=e.size_m, category=e.category)
        return None

    def tag(self, frame, boxes):
        """boxes: [(x1, y1, x2, y2, label, conf)] -> list parallel to boxes: None or dict(name, prob, level, size_m, category)."""
        self.n += 1
        if len(self.mem) == 0:
            return [None] * len(boxes)
        if self.every == 1 or self.n % self.every == 1:
            H, W = frame.shape[:2]
            order = sorted((i for i, b in enumerate(boxes) if b[4] != "person"), key=lambda i: -boxes[i][5])
            regions = [tuple(boxes[i][:4]) for i in order[:self.max_boxes]]
            cb = center_box(W, H, 0.5)
            embs = self.emb.embed_bgr([crop_box(frame, r) for r in regions] + [crop_box(frame, cb)])
            matches = [self.mem.query(e) for e in embs]
            self._cache = [(r, self._extra(m)) for r, m in zip(regions, matches[:-1])]
            mc = matches[-1]
            self.centre_hit = (cb, self._extra(mc)) if (mc.name and mc.level == "high") else None
        out = []
        for b in boxes:
            best, best_iou = None, 0.4
            for r, ex in self._cache:
                v = box_iou(b[:4], r)
                if v > best_iou:
                    best, best_iou = ex, v
            out.append(best)
        return out
