"""End-to-end pipeline test with a fake embedder + fake detector (no torch needed).

Objects are saturated colour squares on a noisy gray background; the 'embedding'
is a hue histogram of saturated pixels, so identity = colour.  This tests the
pipeline LOGIC (enrol -> identify -> find -> scene -> speech), not the neural net.
"""
import cv2
import numpy as np
import pytest

from src.memory import PersonalMemory, l2n
from src.pipeline import Sahayak
from src.schema import Det

W, H = 640, 480
RED, BLUE, GREEN = (30, 30, 220), (220, 60, 30), (40, 200, 40)      # BGR


def make_frame(objs, seed=0, bg_noise=18):
    """objs: list of (bgr_colour, cx, cy, side)."""
    rng = np.random.default_rng(seed)
    f = np.full((H, W, 3), 128, np.uint8)
    f = np.clip(f + rng.normal(0, bg_noise, f.shape), 0, 255).astype(np.uint8)
    for col, cx, cy, s in objs:
        x1, y1 = int(cx - s / 2), int(cy - s / 2)
        patch = np.clip(np.array(col, np.float32) + rng.normal(0, 12, (s, s, 3)), 0, 255)
        f[y1:y1 + s, x1:x1 + s] = patch.astype(np.uint8)
    return f


class FakeEmbedder:
    out_dim = 16

    def embed_bgr(self, imgs):
        out = []
        for im in imgs:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1] > 100
            h = np.zeros(16, np.float32)
            if sat.any():
                hist, _ = np.histogram(hsv[..., 0][sat], bins=16, range=(0, 180))
                h = hist.astype(np.float32)
            out.append(l2n(h + 1e-4))
        return np.array(out, np.float32)


class FakeDetector:
    """Finds saturated blobs and calls them 'bottle'."""

    def __init__(self, label="bottle", enabled=True):
        self.label = label
        self.enabled = enabled

    def detect(self, bgr, classes=None):
        if not self.enabled:
            return []
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 1] > 100).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        dets = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a > 400:
                dets.append(Det((float(x), float(y), float(x + w), float(y + h)), self.label, 0.9))
        return dets


def make_system(detector_on=True):
    mem = PersonalMemory(min_threshold=0.3)
    cfg = {"enroll": {"frames_to_sample": 20, "keep_top_k": 12, "min_sharpness": 10.0},
           "depth": {"hfov_deg": 65.0}}
    return Sahayak(FakeEmbedder(), mem, FakeDetector(enabled=detector_on), None, None, cfg)


def video_of(col, n=14, side=140):
    rng = np.random.default_rng(1)
    return [make_frame([(col, 320 + int(rng.integers(-20, 20)),
                         240 + int(rng.integers(-20, 20)), side)], seed=i) for i in range(n)]


@pytest.fixture
def system():
    s = make_system()
    s.enroll_from_frames("keys", video_of(RED), size_cm=10)
    s.enroll_from_frames("wallet", video_of(BLUE), size_cm=11)
    return s


def test_enroll_records_category_size_and_background(system):
    e = system.mem.entries["keys"]
    assert e.category == "bottle" and abs(e.size_m - 0.10) < 1e-9
    assert len(e.X) >= 5 and len(system.mem.background) > 0


def test_identify_held_known_and_unknown(system):
    f = make_frame([(RED, 320, 240, 140)], seed=99)
    assert "keys" in system.answer_identify(f)
    f2 = make_frame([(GREEN, 320, 240, 140)], seed=98)
    s2 = system.answer_identify(f2)
    assert "keys" not in s2 and "wallet" not in s2


def test_find_personal_left_and_right(system):
    f = make_frame([(RED, 130, 240, 120), (BLUE, 520, 240, 120)], seed=5)
    a = system.respond(f, "where is my keys")
    b = system.respond(f, "where is my wallet")
    assert "left" in a and "keys" in a.lower()
    assert "right" in b and "wallet" in b.lower()


def test_find_has_distance_phrase(system):
    f = make_frame([(RED, 320, 240, 100)], seed=6)
    a = system.respond(f, "where are my keys")
    assert "metre" in a or "arm" in a


def test_find_not_present(system):
    f = make_frame([(GREEN, 320, 240, 120)], seed=7)
    assert "cannot see" in system.respond(f, "where is my keys")


def test_detector_free_fallback_finds_keys():
    s = make_system(detector_on=False)                # detector sees nothing at all
    s.enroll_from_frames("keys", video_of(RED), size_cm=10)
    s.enroll_from_frames("wallet", video_of(BLUE), size_cm=11)
    f = make_frame([(RED, 130, 240, 120), (BLUE, 520, 240, 120)], seed=8)
    obs = s.find_personal(f, "keys")
    assert obs and obs[0].source == "grid"
    cx = (obs[0].box[0] + obs[0].box[2]) / 2
    assert cx < 320                                   # found on the left half


def test_coco_find_and_count(system):
    f = make_frame([(RED, 130, 240, 120), (GREEN, 520, 240, 80)], seed=9)
    a = system.respond(f, "where is the bottle")
    assert "total" in a and ("left" in a or "right" in a)


def test_scene_mentions_personal_names(system):
    f = make_frame([(RED, 130, 240, 120), (BLUE, 520, 240, 120)], seed=10)
    s = system.respond(f, "what's around me")
    assert "your keys" in s and "your wallet" in s


def test_list_forget_unknown_command(system):
    f = make_frame([], seed=11)
    assert "keys" in system.respond(f, "what do you know")
    assert "forgotten" in system.respond(f, "forget keys")
    assert "keys" not in system.mem.names()
    assert "did not understand" in system.respond(f, "banana split sundae")


def test_unknown_object_offers_to_learn(system):
    f = make_frame([], seed=12)
    assert "save this as" in system.respond(f, "where is my spectacles")


def test_too_few_frames_raises():
    s = make_system()
    with pytest.raises(ValueError):
        s.enroll_from_frames("x", video_of(RED, n=2))
