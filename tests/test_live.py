from app.live import LiveApp
from tests.test_pipeline import BLUE, GREEN, RED, make_frame, make_system, video_of


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, key=None, urgent=False, force=False):
        self.said.append(text)
        return True


def build():
    s = make_system()
    s.speaker = FakeSpeaker()
    s.enroll_from_frames("keys", video_of(RED), size_cm=10)
    s.enroll_from_frames("wallet", video_of(BLUE), size_cm=11)
    return LiveApp(s, stride=1), s.speaker


def test_announces_once_then_on_change_and_not_for_unknown():
    app, spk = build()
    for i in range(10):
        app.passive(make_frame([(RED, 320, 240, 140)], seed=100 + i))
    assert sum("keys" in t for t in spk.said) == 1           # announced once, not every frame
    for i in range(10):
        app.passive(make_frame([(BLUE, 320, 240, 140)], seed=200 + i))
    assert sum("wallet" in t for t in spk.said) == 1
    n = len(spk.said)
    for i in range(10):
        app.passive(make_frame([(GREEN, 320, 240, 140)], seed=300 + i))
    assert len(spk.said) == n                                  # unknown object: stays silent


def test_empty_memory_is_silent():
    s = make_system()
    s.speaker = FakeSpeaker()
    app = LiveApp(s, stride=1)
    for i in range(5):
        app.passive(make_frame([(RED, 320, 240, 140)], seed=i))
    assert s.speaker.said == []
