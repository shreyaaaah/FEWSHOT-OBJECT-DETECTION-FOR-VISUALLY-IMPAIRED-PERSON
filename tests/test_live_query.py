import live

NAMES = ["person", "laptop", "cell phone", "cup", "bottle", "toothbrush", "teddy bear", "dining table"]


def item(label, side="slightly to your right", dist=1.0, reliable=True, conf=0.8):
    return dict(label=label, conf=conf, box=(0, 0, 1, 1), side=side, dist=dist, reliable=reliable, level="high")


def test_find_target():
    f = lambda t: live.find_target(t, NAMES)
    assert f("Where is my laptop?") == "laptop"
    assert f("where is my phone") == "cell phone"
    assert f("find my mug please") == "cup"
    assert f("is there a teddy bear") == "teddy bear"
    assert f("where are my bottles") == "bottle"
    assert f("hello there") is None


def test_answer_with_direction_and_distance():
    t = live.answer("laptop", [item("laptop")])
    assert t == "The laptop is slightly to your right, about one metre away."
    assert live.answer("laptop", [item("cup")]) is None
    assert "2 laptops" in live.answer("laptop", [item("laptop"), item("laptop", side="to your left")])
    assert live.answer("laptop", [item("laptop", dist=None)]) == "The laptop is slightly to your right."


def test_keep_is_stricter_for_person():
    assert not live.keep("person", 0.45, 0.30)
    assert live.keep("person", 0.60, 0.30)
    assert live.keep("cup", 0.35, 0.30) and not live.keep("cup", 0.25, 0.30)


def test_segmenter_cuts_one_phrase_and_ignores_clicks_and_own_voice():
    import numpy as np
    rng = np.random.default_rng(0)
    quiet = lambda: (rng.standard_normal(1600) * 40).astype(np.int16)
    speech = lambda: (rng.standard_normal(1600) * 3000).astype(np.int16)
    seg = live.Segmenter()
    out = []
    for blk in [quiet()] * 15 + [speech()] * 12 + [quiet()] * 12:
        r = seg.feed(blk)
        if r:
            out.append(r)
    assert len(out) == 1 and 12 * 1600 * 2 <= len(out[0]) <= 24 * 1600 * 2
    seg = live.Segmenter()                                    # one short click is not a phrase
    assert [seg.feed(b) for b in [quiet()] * 10 + [speech()] + [quiet()] * 15] == [None] * 26
    seg = live.Segmenter()                                    # muted while Sahayak speaks
    assert all(seg.feed(b, busy=True) is None for b in [speech()] * 30)


# ---------------------------------------------------------------- personal objects in the live view
def test_find_target_prefers_taught_name_over_synonym():
    names = NAMES + ["mobile", "shreya's mobile"]
    assert live.find_target("where is my mobile", names) == "mobile"
    assert live.find_target("where is shreyas mobile", names) == "shreya's mobile"
    assert live.find_target("where is my phone", names) == "cell phone"      # no taught 'phone' -> generic phone


def test_two_phones_one_taught_gets_its_own_name():
    import numpy as np
    from src.personal_scene import PersonalTagger
    from src.schema import Match

    class Emb:                      # bright crop -> [1, 0], dark crop -> [0, 1]
        def embed_bgr(self, imgs):
            return np.array([[1.0, 0.0] if im.mean() > 128 else [0.0, 1.0] for im in imgs], np.float32)

    class Entry:
        size_m, category = None, "cell phone"

    class Mem:
        entries = {"mobile": Entry()}

        def __len__(self):
            return 1

        def query(self, e):
            if e[0] > 0.5:
                return Match("mobile", "mobile", 0.9, 0.6, 0.3, None, 0.0, "high", 0.9)
            return Match(None, None, 0.1, 0.6, 0.0, None, 0.0, "unknown", 0.1)

    frame = np.zeros((240, 320, 3), np.uint8)
    frame[60:120, 20:100] = 255                                   # left phone: bright (the taught one)
    frame[60:120, 200:280] = 30                                   # right phone: dark (a normal phone)
    boxes = [(20, 60, 100, 120, "cell phone", 0.8), (200, 60, 280, 120, "cell phone", 0.8)]
    extras = PersonalTagger(Emb(), Mem(), every=1).tag(frame, boxes)
    assert extras[0]["name"] == "mobile" and extras[1] is None
    items = live.analyze(boxes, 320, 240, 65.0, extras)
    names = sorted(i["label"] for i in items)
    assert names == ["cell phone", "mobile"]
    text = live.summary(items)
    assert "your mobile" in text and "a cell phone" in text
    mobile = [i for i in items if i["label"] == "mobile"][0]
    assert mobile["personal"] and not mobile["reliable"]          # size only guessed from the class -> 'roughly'
    assert live.answer("mobile", items).startswith("Your mobile is ")
