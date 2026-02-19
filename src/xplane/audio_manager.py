"""
Audio Manager for Voicemeeter output routing.

Routes server-side beep/TTS to a specific Voicemeeter virtual input strip
via sounddevice. Mic input is handled by Chrome's Web Speech API directly
(user sets Chrome's mic to Voicemeeter B1 in chrome://settings/content/microphone).
"""

import logging
import threading
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_sd = None
_np = None
_pyttsx3 = None


def _import_sounddevice():
    global _sd
    if _sd is None:
        import sounddevice as sd
        _sd = sd
    return _sd


def _import_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _import_pyttsx3():
    global _pyttsx3
    if _pyttsx3 is None:
        import pyttsx3
        _pyttsx3 = pyttsx3
    return _pyttsx3


class AudioManager:
    """Manages Voicemeeter output routing for beep/TTS."""

    _PREFERRED_HOSTAPI = "Windows WASAPI"

    _VM_OUTPUT_PATTERNS = [
        ("Voicemeeter Input",       "Voicemeeter Input"),
        ("Voicemeeter AUX Input",   "Voicemeeter AUX Input"),
        ("Voicemeeter VAIO3 Input", "Voicemeeter VAIO3"),
    ]

    def __init__(self):
        self._output_device_name: str = ""
        self._output_device_index: Optional[int] = None
        self._confirmation_beep: bool = True
        self._confirmation_tts: bool = True
        self._playback_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize audio subsystem."""
        try:
            _import_sounddevice()
            _import_numpy()
            self._initialized = True
            logger.info("AudioManager initialized")
            return True
        except ImportError as e:
            logger.error(f"AudioManager init failed: {e}")
            return False

    def configure(self, output_device_name: str = "",
                  confirmation_beep: bool = True,
                  confirmation_tts: bool = True, **kwargs):
        """Update output configuration."""
        if output_device_name != self._output_device_name:
            self._output_device_name = output_device_name
            self._output_device_index = self._resolve_device_index(
                output_device_name, "output")
        self._confirmation_beep = confirmation_beep
        self._confirmation_tts = confirmation_tts

    # ------------------------------------------------------------------
    # Device Enumeration
    # ------------------------------------------------------------------

    def list_devices(self) -> dict:
        """List Voicemeeter output devices."""
        sd = _import_sounddevice()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        preferred_api = None
        for idx, api in enumerate(hostapis):
            if api["name"] == self._PREFERRED_HOSTAPI:
                preferred_api = idx
                break

        output_devices = []
        for i, dev in enumerate(devices):
            if preferred_api is not None and dev["hostapi"] != preferred_api:
                continue
            name = dev["name"]
            if dev["max_output_channels"] > 0:
                for pattern, label in self._VM_OUTPUT_PATTERNS:
                    if name.startswith(pattern):
                        output_devices.append({
                            "index": i, "name": name, "label": label,
                        })
                        break

        return {"output_devices": output_devices}

    def _resolve_device_index(self, name: str, kind: str = "output") -> Optional[int]:
        """Resolve a device name to its WASAPI index."""
        if not name:
            return None
        try:
            sd = _import_sounddevice()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            wasapi_idx = None
            for idx, api in enumerate(hostapis):
                if api["name"] == self._PREFERRED_HOSTAPI:
                    wasapi_idx = idx
                    break

            for i, dev in enumerate(devices):
                if wasapi_idx is not None and dev["hostapi"] != wasapi_idx:
                    continue
                if dev["name"] == name:
                    if kind == "output" and dev["max_output_channels"] > 0:
                        return i
        except Exception as e:
            logger.error(f"Device resolve error: {e}")
        return None

    # ------------------------------------------------------------------
    # Beep Generation
    # ------------------------------------------------------------------

    def _get_output_samplerate(self) -> int:
        if self._output_device_index is None:
            return 48000
        try:
            sd = _import_sounddevice()
            return int(sd.query_devices(
                self._output_device_index)["default_samplerate"])
        except Exception:
            return 48000

    def play_beep(self, freq: int = 800, duration: float = 0.15,
                  volume: float = 0.3) -> bool:
        """Play a sine-wave beep on the configured output device."""
        with self._playback_lock:
            if self._output_device_index is None:
                return False
            try:
                sd = _import_sounddevice()
                np = _import_numpy()
                sr = self._get_output_samplerate()
                t = np.linspace(0, duration, int(sr * duration),
                                endpoint=False)
                tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
                fade = int(sr * 0.01)
                if fade > 0 and len(tone) > 2 * fade:
                    tone[:fade] *= np.linspace(0, 1, fade,
                                               dtype=np.float32)
                    tone[-fade:] *= np.linspace(1, 0, fade,
                                                dtype=np.float32)
                tone *= volume
                sd.play(tone, samplerate=sr,
                        device=self._output_device_index, blocking=True)
                return True
            except Exception as e:
                logger.error(f"Beep error: {e}")
                return False

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str) -> bool:
        """Speak text via pyttsx3 TTS on the configured output device."""
        with self._tts_lock:
            if self._output_device_index is None:
                return False
            try:
                import tempfile
                sd = _import_sounddevice()
                np = _import_numpy()
                pyttsx3 = _import_pyttsx3()

                tmp = tempfile.NamedTemporaryFile(suffix=".wav",
                                                  delete=False)
                tmp_path = tmp.name
                tmp.close()

                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                engine.stop()

                with wave.open(tmp_path, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
                    nch = wf.getnchannels()
                    sw = wf.getsampwidth()

                if sw == 2:
                    audio = np.frombuffer(
                        frames, dtype=np.int16
                    ).astype(np.float32) / 32768.0
                elif sw == 1:
                    audio = (np.frombuffer(
                        frames, dtype=np.uint8
                    ).astype(np.float32) - 128) / 128.0
                else:
                    audio = np.frombuffer(
                        frames, dtype=np.int16
                    ).astype(np.float32) / 32768.0

                if nch > 1:
                    audio = audio.reshape(-1, nch)[:, 0]

                dev_sr = self._get_output_samplerate()
                if sr != dev_sr:
                    ratio = dev_sr / sr
                    new_len = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_len)
                    audio = np.interp(
                        indices, np.arange(len(audio)), audio
                    ).astype(np.float32)
                    sr = dev_sr

                sd.play(audio, samplerate=sr,
                        device=self._output_device_index, blocking=True)

                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
                return True
            except Exception as e:
                logger.error(f"TTS error: {e}")
                return False

    def play_confirmation(self, text: str) -> None:
        """Play confirmation beep + TTS in background."""
        def _do():
            if self._confirmation_beep:
                self.play_beep(freq=800, duration=0.15, volume=0.3)
            if self._confirmation_tts and text:
                self.speak(text)
        threading.Thread(target=_do, daemon=True,
                         name="audio-confirm").start()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def server_side_output(self) -> bool:
        """True if an output device is configured for TTS/beep."""
        return bool(self._output_device_name
                    and self._output_device_index is not None)

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "output_device_name": self._output_device_name,
            "output_device_index": self._output_device_index,
            "confirmation_beep": self._confirmation_beep,
            "confirmation_tts": self._confirmation_tts,
        }

    def shutdown(self):
        """Clean up."""
        logger.info("AudioManager shut down")
