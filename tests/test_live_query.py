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
