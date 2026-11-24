from live import Announcer


def it(label, side="straight ahead", dist=1.0, conf=0.7, at=None):
    k = at if at is not None else {"cup": 0, "person": 10}.get(label, 20)
    return dict(label=label, side=side, dist=dist, conf=conf, reliable=True, level="high", box=(k, 0, k + 5, 5))


def run(frames, **kw):
    a, spoken, t = Announcer(**kw), [], 0.0
    for f in frames:
        text, _ = a.update(f, t)
        if text:
            spoken.append((t, text))
        t += 0.2
    return spoken


def test_single_frame_flicker_is_not_announced():
    frames = [[]] * 4 + [[it("cup")]] + [[]] * 6
    assert run(frames) == []


def test_stable_object_announced_once_then_refreshed_late():
    frames = [[it("cup")]] * 20                      # 4 s of video
    sp = run(frames)
    assert len(sp) == 1 and "cup" in sp[0][1]


def test_dropped_frames_do_not_reannounce():
    frames = [[it("cup")]] * 4 + [[]] + [[it("cup")]] * 4 + [[]] + [[it("cup")]] * 4
    assert len(run(frames, min_gap=0.0)) == 1


def test_new_object_is_announced_and_busy_blocks_speech():
    a = Announcer(min_gap=0.0)
    for k in range(4):
        a.update([it("cup")], k * 0.2)
    text, _ = a.update([it("cup"), it("person", dist=2.0)], 1.0)
    text, _ = a.update([it("cup"), it("person", dist=2.0)], 1.2, busy=True)
    assert text is None
    seen = [a.update([it("cup"), it("person", dist=2.0)], 1.4 + k * 0.2)[0] for k in range(5)]
    assert any(t and "person" in t for t in seen)


def test_repeat_while_unchanged():
    a = Announcer(repeat=20.0)
    spoken = [a.update([it("cup")], t)[0] for t in [0.0, 0.2, 0.4, 0.6, 21.0]]
    assert sum(bool(x) for x in spoken) == 2


def test_label_flip_on_same_object_keeps_the_better_label():
    # a mug is first seen as 'cell phone' (conf 0.42) for 2 frames, then as 'cup' (conf 0.7) for 4 frames
    frames = [[it("cell phone", conf=0.42, at=0)]] * 2 + [[it("cup", conf=0.7, at=0)]] * 6
    sp = run(frames)
    assert sp and all("cell phone" not in t for _, t in sp) and "cup" in sp[0][1]
