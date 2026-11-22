"""Queued Windows speech with ordered SAPI, pyttsx3, and PowerShell fallbacks."""
import base64
import queue
import subprocess
import threading
import traceback


class Voice:
    """Speak queued text from a background thread and report backend errors."""

    def __init__(self, rate=170):
        self.rate = rate
        self.backend = "starting"
        self._ready = threading.Event()
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _powershell_speak(self, text):
        encoded_text = base64.b64encode(text.encode("utf-16le")).decode("ascii")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"
            + encoded_text
            + "')); "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Speak($text)"
        )
        encoded_command = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-EncodedCommand", encoded_command],
            check=True,
            capture_output=True,
            text=True,
        )

    def _run(self):
        pythoncom = None
        sapi = None
        engine = None
        try:
            import pythoncom as com_module
            import win32com.client

            pythoncom = com_module
            pythoncom.CoInitialize()
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
            self.backend = "win32com.client SAPI.SpVoice"
        except Exception:
            print("SAPI initialization failed:")
            traceback.print_exc()
        if sapi is None:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                self.backend = "pyttsx3"
            except Exception:
                print("pyttsx3 initialization failed:")
                traceback.print_exc()
        if sapi is None and engine is None:
            self.backend = "PowerShell System.Speech"
        self._ready.set()

        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            text, done = item
            try:
                if sapi is not None:
                    try:
                        sapi.Speak(text)
                    except Exception:
                        print("SAPI speech failed:")
                        traceback.print_exc()
                        sapi = None
                        try:
                            import pyttsx3

                            engine = pyttsx3.init()
                            engine.setProperty("rate", self.rate)
                            self.backend = "pyttsx3"
                        except Exception:
                            print("pyttsx3 fallback initialization failed:")
                            traceback.print_exc()
                            engine = None
                if sapi is None and engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        print("pyttsx3 speech failed:")
                        traceback.print_exc()
                        engine = None
                if sapi is None and engine is None:
                    try:
                        self._powershell_speak(text)
                        self.backend = "PowerShell System.Speech"
                    except Exception:
                        print("PowerShell speech failed:")
                        traceback.print_exc()
                        self.backend = "unavailable"
            finally:
                done.set()
                self._queue.task_done()
        if pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                print("COM cleanup failed:")
                traceback.print_exc()

    def speak(self, text):
        """Queue speech and return an event set when this text has been handled."""
        done = threading.Event()
        self._queue.put((str(text), done))
        return done

    def wait_ready(self, timeout=None):
        return self._ready.wait(timeout)

    def clear_pending(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                if item is not None:
                    item[1].set()
                self._queue.task_done()

    def close(self):
        self._queue.put(None)

