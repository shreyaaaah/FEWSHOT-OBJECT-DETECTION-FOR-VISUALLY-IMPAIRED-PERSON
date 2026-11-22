"""Audio feedback: non-blocking text-to-speech with cooldown + earcons."""
import queue
import sys
import threading
import time


class Speaker:
    """Speaks in a background thread (pyttsx3 = offline, works on Win/Mac/Linux).
    If pyttsx3 is not available it prints, so the project still runs anywhere."""

    def __init__(self, enabled=True, cooldown=2.5, rate=170, verbose=True):
        self.enabled = enabled
        self.cooldown = cooldown
        self.rate = rate
        self.verbose = verbose
        self.q = queue.Queue()
        self._last = {}
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def say(self, text: str, key=None, urgent=False, force=False) -> bool:
        """Queue speech. Same `key` is suppressed for `cooldown` seconds."""
        if not text:
            return False
        key = key or text
        now = time.time()
        if not force and now - self._last.get(key, 0.0) < self.cooldown:
            return False
        self._last[key] = now
        if urgent:
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(text)
        return True

    def _run(self):
        engine = None
        if self.enabled:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
            except Exception:
                engine = None
        while True:
            text = self.q.get()
            if text is None:
                break
            if self.verbose:
                print(f"[audio] {text}")
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass

    def close(self):
        self.q.put(None)


def beep(kind="ok"):
    """Short earcon: 'start' = recording started, 'ok' = saved, 'err' = failed."""
    freqs = {"start": (880, 120), "ok": (1200, 200), "err": (300, 400)}
    f, ms = freqs.get(kind, (1000, 150))
    try:
        import winsound
        winsound.Beep(f, ms)
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()
