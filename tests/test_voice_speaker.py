from src.speaker import VoiceSpeaker


class FakeVoice:
    backend = "fake"

    def __init__(self):
        self.spoken, self.cleared = [], 0

    def speak(self, t):
        self.spoken.append(t)

    def clear_pending(self):
        self.cleared += 1

    def close(self):
        pass


def test_speaks_and_respects_cooldown_force_urgent():
    v = FakeVoice()
    s = VoiceSpeaker(cooldown=100, verbose=False, voice=v)
    assert s.say("hello") and not s.say("hello")          # same key inside the cooldown is suppressed
    assert s.say("hello", force=True)                      # force overrides
    assert s.say("stop", urgent=True) and v.cleared == 1   # urgent drops the waiting queue first
    assert v.spoken == ["hello", "hello", "stop"]
    assert not s.say("")


def test_disabled_speaker_makes_no_voice():
    s = VoiceSpeaker(enabled=False, verbose=False)
    assert s.say("x") and s.backend == "off"
