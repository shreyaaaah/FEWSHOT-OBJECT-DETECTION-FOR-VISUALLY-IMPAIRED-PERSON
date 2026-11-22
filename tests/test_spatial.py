from src.spatial import (clock_hour, describe_location, distance_bin, distance_phrase,
                         framing_hint, guidance, horizontal_angle, scene_phrase,
                         side_phrase)


def test_angle_sign_and_zero():
    assert abs(horizontal_angle(320, 640, 65)) < 1e-6
    assert horizontal_angle(100, 640, 65) < 0
    assert horizontal_angle(560, 640, 65) > 0


def test_clock_hours():
    assert clock_hour(0) == 12
    assert clock_hour(30) == 1
    assert clock_hour(-30) == 11
    assert clock_hour(90) == 3
    assert clock_hour(-90) == 9


def test_side_phrase():
    assert side_phrase(0, 65) == "straight ahead"
    assert "left" in side_phrase(-25, 65) and "slightly" not in side_phrase(-25, 65)
    assert side_phrase(8, 65) == "slightly to your right"


def test_distance_bins():
    assert distance_bin(0.3) == 0
    assert distance_bin(1.0) == 2
    assert distance_bin(10) == 5
    assert distance_bin(None) is None
    assert "arm" in distance_phrase(0.2)
    assert distance_phrase(1.0, reliable=False).startswith("roughly")
    assert "more than" in distance_phrase(9)


def test_describe_location_left():
    s = describe_location("Your bottle", (50, 200, 150, 300), 640, 480, 1.0)
    assert "left" in s and "one metre" in s and s.endswith(".")


def test_describe_location_centre_no_clock():
    s = describe_location("The cup", (300, 200, 340, 260), 640, 480, 0.3)
    assert "straight ahead" in s and "o'clock" not in s and "arm" in s


def test_guidance():
    assert guidance((20, 200, 80, 260), 640, 480) == "Turn left."
    assert guidance((560, 200, 620, 260), 640, 480) == "Turn right."
    assert "Reach" in guidance((300, 200, 340, 260), 640, 480, dist_m=0.3)
    assert "Move forward" in guidance((300, 200, 340, 260), 640, 480, dist_m=2.0)


def test_framing_hint():
    assert "closer" in framing_hint((300, 220, 320, 240), 640, 480)
    assert "farther" in framing_hint((0, 0, 620, 470), 640, 480)
    assert framing_hint((200, 120, 440, 360), 640, 480) is None


def test_scene_phrase_sorted_nearest_first():
    items = [("cup", (50, 100, 100, 150), 2.0, True),
             ("your keys", (300, 100, 340, 140), 0.4, True)]
    s = scene_phrase(items, 640, 480)
    assert s.index("keys") < s.index("cup")
    assert scene_phrase([], 640, 480).startswith("I do not")


def test_verb_agreement():
    from src.spatial import verb_for
    assert verb_for("Your keys") == "are" and verb_for("I think your spectacles") == "are"
    assert verb_for("Your bottle") == "is" and verb_for("The glass") == "is"
    assert "keys are" in describe_location("Your keys", (50, 200, 150, 300), 640, 480, 1.0)
