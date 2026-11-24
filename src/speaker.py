"""Speaker for the personal-object app, built on the voice that is known to work on Windows (src/tts.Voice:
SAPI, then pyttsx3, then PowerShell). Same interface as src.audio.Speaker, but it never goes silent just
because pyttsx3 is missing."""
import time

from .tts import Voice


class VoiceSpeaker:
    def __init__(self, enabled=True, cooldown=2.5, verbose=True, voice=None):
        self.enabled, self.cooldown, self.verbose = enabled, cooldown, verbose
        self._last = {}
        self.voice = voice if voice is not None else (Voice() if enabled else None)

    @property
    def backend(self):
        return self.voice.backend if self.voice else "off"

    def say(self, text, key=None, urgent=False, force=False):
        """Speak `text`. The same `key` is suppressed for `cooldown` seconds unless force=True;
        urgent=True drops anything still waiting to be spoken."""
        if not text:
            return False
        key = key or text
        now = time.time()
        if not force and now - self._last.get(key, 0.0) < self.cooldown:
            return False
        self._last[key] = now
        if self.verbose:
            print(f"[audio] {text}")
        if self.voice is not None:
            if urgent:
                self.voice.clear_pending()
            self.voice.speak(text)
        return True

    def close(self):
        if self.voice is not None:
            self.voice.close()
