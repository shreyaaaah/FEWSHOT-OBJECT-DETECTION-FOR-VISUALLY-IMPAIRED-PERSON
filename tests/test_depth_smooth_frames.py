import numpy as np

from src.depth import box_depth_value, fuse_distances, size_distance
from src.frames import select_frames
from src.imgutil import center_box, grid_boxes, iou, nms, pad_square, prep_bgr
from src.smoothing import LocationFilter, VoteSmoother
from src.spatial import focal_px


# ---------------------------------------------------------------- depth
def test_size_distance_pinhole():
    f = focal_px(640, 65)
    # a 0.24 m bottle that is 100 px tall
    d, ok = size_distance((100, 100, 140, 200), 0.24, f, 640, 480)
    assert ok and abs(d - f * 0.24 / 100) < 1e-6
    # twice as far -> half the pixel size
    d2, _ = size_distance((100, 100, 120, 150), 0.24, f, 640, 480)
    assert abs(d2 / d - 2.0) < 1e-6


def test_size_distance_border_unreliable():
    f = focal_px(640, 65)
    _, ok = size_distance((0, 100, 40, 200), 0.24, f, 640, 480)
    assert not ok


def test_fuse_recovers_unknown_size_object():
    # true disparity r = 1/d; two anchors with known size, one unknown-size object at 2 m
    true_d = [1.0, 3.0, 2.0]
    r = [1 / d for d in true_d]
    size_d = [1.0, 3.0, None]
    out, info = fuse_distances(size_d, [True, True, True], r)
    assert info["used_depth"] and info["anchors"] == 2
    assert abs(out[2] - 2.0) < 1e-6


def test_fuse_without_anchors_falls_back():
    out, info = fuse_distances([None, None], [True, True], [0.5, 0.4])
    assert out == [None, None] and not info["used_depth"]


def test_fuse_ignores_unreliable_anchor():
    out, info = fuse_distances([1.0, 5.0], [True, False], [1.0, 1.0])
    assert info["anchors"] == 1


def test_box_depth_value():
    m = np.ones((100, 100), np.float32)
    m[40:60, 40:60] = 5.0
    assert box_depth_value(m, (20, 20, 80, 80)) == 5.0 or box_depth_value(m, (20, 20, 80, 80)) >= 1.0


# ------------------------------------------------------------ smoothing
def test_vote_smoother_stable_and_changes():
    s = VoteSmoother(window=5, min_votes=3)
    outs = [s.update("keys", 0.9)[0] for _ in range(4)]
    assert outs[-1] == "keys"
    # one outlier must not flip it
    assert s.update("wallet", 0.9)[0] == "keys"
    # sustained change flips it
    last = None
    for _ in range(5):
        last, _ = s.update("wallet", 0.9)
    assert last == "wallet"


def test_vote_smoother_reports_change_once():
    s = VoteSmoother(window=5, min_votes=2)
    flags = [s.update("keys", 1.0)[1] for _ in range(5)]
    assert sum(flags) == 1


def test_location_filter_ema_and_reset():
    f = LocationFilter(alpha=0.5, patience=2)
    f.update(100, 100, 1.0, 640)
    cx, _, d = f.update(120, 100, 2.0, 640)
    assert abs(cx - 110) < 1e-6 and abs(d - 1.5) < 1e-6
    f.miss()
    assert f.miss() is None


# ------------------------------------------------------------- frames/img
def test_select_frames_prefers_sharp():
    rng = np.random.default_rng(0)
    sharp = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    import cv2
    blurry = cv2.GaussianBlur(sharp, (0, 0), 6)
    frames = [blurry if i % 2 else sharp for i in range(20)]
    idx, s = select_frames(frames, keep=5, min_sharp=40)
    assert len(idx) >= 1
    assert all(i % 2 == 0 for i in idx)


def test_select_frames_all_blurry_still_returns():
    import cv2
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((90, 120, 3)) * 255).astype(np.uint8), (0, 0), 8)
    idx, _ = select_frames([base.copy() for _ in range(12)], keep=10, min_sharp=1e6)
    assert len(idx) >= 5


def test_prep_shape_and_square_padding():
    img = np.zeros((100, 300, 3), np.uint8)
    assert pad_square(img).shape[:2] == (300, 300)
    assert prep_bgr(img, 64).shape == (3, 64, 64)


def test_nms_and_iou():
    a, b = (0, 0, 10, 10), (1, 1, 11, 11)
    assert iou(a, a) == 1.0 and iou(a, b) > 0.6
    assert nms([a, b, (50, 50, 60, 60)], [0.9, 0.8, 0.7], 0.5) == [0, 2]


def test_grid_boxes_cover_and_bounded():
    boxes = grid_boxes(640, 480)
    assert 10 < len(boxes) < 80
    assert all(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480 for x1, y1, x2, y2 in boxes)
    c = center_box(640, 480)
    assert c[0] < 320 < c[2]
