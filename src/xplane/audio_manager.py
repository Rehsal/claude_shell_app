"""
Audio Manager for Voicemeeter WASAPI audio routing.

Provides server-side mic capture, Vosk speech recognition,
beep generation, and TTS playback via specific Voicemeeter devices.
Falls back to browser Web Speech API when no devices are configured.
"""

import io
import logging
import math
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_sd = None
_np = None
_vosk = None
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


def _import_vosk():
    global _vosk
    if _vosk is None:
        import vosk
        vosk.SetLogLevel(-1)
        _vosk = vosk
    return _vosk


def _import_pyttsx3():
    global _pyttsx3
    if _pyttsx3 is None:
        import pyttsx3
        _pyttsx3 = pyttsx3
    return _pyttsx3


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)


class AudioManager:
    """Manages WASAPI audio capture/playback through Voicemeeter devices."""

    def __init__(self):
        self._mic_device_name: str = ""
        self._mic_device_index: Optional[int] = None
        self._output_device_name: str = ""
        self._output_device_index: Optional[int] = None
        self._vosk_model_path: str = "data/vosk-model-small-en-us-0.15"
        self._vosk_model = None
        self._confirmation_beep: bool = True
        self._confirmation_tts: bool = True

        # Capture state
        self._capture_lock = threading.Lock()
        self._capturing = False
        self._capture_stream = None
        self._capture_frames: list = []
        self._mic_rms: float = 0.0

        # Playback / TTS locks
        self._tts_lock = threading.Lock()
        self._playback_lock = threading.Lock()

        # TTS engine (lazy)
        self._tts_engine = None
        self._tts_engine_lock = threading.Lock()

        self._initialized = False

    def initialize(self) -> bool:
        """Initialize audio subsystem. Returns True if at least sounddevice loads."""
        try:
            _import_sounddevice()
            _import_numpy()
            self._initialized = True
            logger.info("AudioManager initialized (sounddevice + numpy OK)")
        except ImportError as e:
            logger.error(f"AudioManager init failed: {e}")
            return False

        # Try loading vosk model
        try:
            self._load_vosk_model()
        except Exception as e:
            logger.warning(f"Vosk model not available: {e}")

        return True

    def _load_vosk_model(self):
        """Load Vosk model from configured path."""
        model_path = Path(self._vosk_model_path)
        if not model_path.exists():
            logger.warning(f"Vosk model not found at {model_path}")
            return
        vosk = _import_vosk()
        self._vosk_model = vosk.Model(str(model_path))
        logger.info(f"Vosk model loaded from {model_path}")

    def configure(self, mic_device_name: str = "", output_device_name: str = "",
                  vosk_model_path: str = "", confirmation_beep: bool = True,
                  confirmation_tts: bool = True):
        """Update device configuration. Re-resolves device indices by name."""
        if mic_device_name != self._mic_device_name:
            self._mic_device_name = mic_device_name
            self._mic_device_index = self._resolve_device_index(mic_device_name, kind="input")

        if output_device_name != self._output_device_name:
            self._output_device_name = output_device_name
            self._output_device_index = self._resolve_device_index(output_device_name, kind="output")

        if vosk_model_path and vosk_model_path != self._vosk_model_path:
            self._vosk_model_path = vosk_model_path
            self._vosk_model = None
            try:
                self._load_vosk_model()
            except Exception as e:
                logger.warning(f"Could not load vosk model: {e}")

        self._confirmation_beep = confirmation_beep
        self._confirmation_tts = confirmation_tts

    @property
    def server_side_active(self) -> bool:
        """True if a mic device is configured (server-side audio mode)."""
        return bool(self._mic_device_name and self._mic_device_index is not None)

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    def list_devices(self, filter_voicemeeter: bool = True) -> dict:
        """List available audio devices. Optionally filter for Voicemeeter only."""
        sd = _import_sounddevice()
        devices = sd.query_devices()
        input_devices = []
        output_devices = []

        for i, dev in enumerate(devices):
            name = dev["name"]
            if filter_voicemeeter and "voicemeeter" not in name.lower():
                continue

            entry = {"index": i, "name": name, "hostapi": dev["hostapi"]}

            if dev["max_input_channels"] > 0:
                input_devices.append(entry)
            if dev["max_output_channels"] > 0:
                output_devices.append(entry)

        return {"input_devices": input_devices, "output_devices": output_devices}

    def _resolve_device_index(self, name: str, kind: str = "input") -> Optional[int]:
        """Resolve a device name to its index. Returns None if not found."""
        if not name:
            return None
        try:
            sd = _import_sounddevice()
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["name"] == name:
                    if kind == "input" and dev["max_input_channels"] > 0:
                        return i
                    if kind == "output" and dev["max_output_channels"] > 0:
                        return i
        except Exception as e:
            logger.error(f"Device resolve error: {e}")
        return None

    # ------------------------------------------------------------------
    # WASAPI Capture
    # ------------------------------------------------------------------

    def start_capture(self) -> bool:
        """Start capturing audio from the configured mic device."""
        with self._capture_lock:
            if self._capturing:
                return True
            if self._mic_device_index is None:
                logger.warning("No mic device configured")
                return False

            sd = _import_sounddevice()
            np = _import_numpy()
            self._capture_frames = []
            self._mic_rms = 0.0

            try:
                def callback(indata, frames, time_info, status):
                    if status:
                        logger.debug(f"Capture status: {status}")
                    self._capture_frames.append(indata.copy())
                    # Update RMS level
                    rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
                    # Normalize int16 RMS to 0-1
                    self._mic_rms = min(1.0, rms / 3000.0)

                self._capture_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    device=self._mic_device_index,
                    channels=CHANNELS,
                    dtype="int16",
                    callback=callback,
                )
                self._capture_stream.start()
                self._capturing = True
                logger.info(f"Capture started on device {self._mic_device_index}")
                return True
            except Exception as e:
                logger.error(f"Capture start failed: {e}")
                self._capturing = False
                return False

    def stop_capture(self) -> bytes:
        """Stop capture and return WAV bytes."""
        with self._capture_lock:
            if not self._capturing:
                return b""

            try:
                if self._capture_stream:
                    self._capture_stream.stop()
                    self._capture_stream.close()
            except Exception as e:
                logger.error(f"Capture stop error: {e}")
            finally:
                self._capture_stream = None
                self._capturing = False

            if not self._capture_frames:
                return b""

            np = _import_numpy()
            audio_data = np.concatenate(self._capture_frames, axis=0)
            self._capture_frames = []
            self._mic_rms = 0.0

            # Convert to WAV bytes
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # int16
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())
            return buf.getvalue()

    def get_mic_level(self) -> float:
        """Get current mic RMS level (0.0 - 1.0)."""
        return self._mic_rms

    # ------------------------------------------------------------------
    # Vosk Recognition
    # ------------------------------------------------------------------

    def recognize(self, wav_bytes: bytes) -> Optional[str]:
        """Recognize speech from WAV bytes using Vosk. Returns text or None."""
        if not self._vosk_model:
            logger.warning("Vosk model not loaded — cannot recognize")
            return None
        if not wav_bytes:
            return None

        try:
            vosk = _import_vosk()
            import json

            wf = wave.open(io.BytesIO(wav_bytes), "rb")
            rec = vosk.KaldiRecognizer(self._vosk_model, wf.getframerate())
            rec.SetWords(False)

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            wf.close()
            return text if text else None
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None

    # ------------------------------------------------------------------
    # Beep Generation
    # ------------------------------------------------------------------

    def play_beep(self, freq: int = 800, duration: float = 0.15,
                  volume: float = 0.3) -> bool:
        """Play a sine-wave beep on the configured output device."""
        with self._playback_lock:
            if self._output_device_index is None:
                return False

            try:
                sd = _import_sounddevice()
                np = _import_numpy()

                t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
                tone = np.sin(2 * np.pi * freq * t).astype(np.float32)

                # Fade envelope (10ms fade in/out)
                fade_samples = int(SAMPLE_RATE * 0.01)
                if fade_samples > 0 and len(tone) > 2 * fade_samples:
                    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
                    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
                    tone[:fade_samples] *= fade_in
                    tone[-fade_samples:] *= fade_out

                tone *= volume
                sd.play(tone, samplerate=SAMPLE_RATE, device=self._output_device_index,
                        blocking=True)
                return True
            except Exception as e:
                logger.error(f"Beep playback error: {e}")
                return False

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _get_tts_engine(self):
        """Get or create the pyttsx3 TTS engine (must be used from one thread)."""
        with self._tts_engine_lock:
            if self._tts_engine is None:
                pyttsx3 = _import_pyttsx3()
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 170)
            return self._tts_engine

    def speak(self, text: str) -> bool:
        """Speak text via TTS on the configured output device."""
        with self._tts_lock:
            if self._output_device_index is None:
                return False

            try:
                import tempfile
                sd = _import_sounddevice()
                np = _import_numpy()

                # Render TTS to WAV file
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()

                engine = self._get_tts_engine()
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()

                # Read and play the WAV
                with wave.open(tmp_path, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
                    nch = wf.getnchannels()
                    sw = wf.getsampwidth()

                # Convert to float32 for playback
                if sw == 2:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sw == 1:
                    audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
                else:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                if nch > 1:
                    audio = audio.reshape(-1, nch)[:, 0]  # mono

                sd.play(audio, samplerate=sr, device=self._output_device_index,
                        blocking=True)

                # Clean up temp file
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

                return True
            except Exception as e:
                logger.error(f"TTS error: {e}")
                return False

    def play_confirmation(self, text: str) -> None:
        """Play confirmation beep + TTS in background (fire-and-forget)."""
        def _do():
            if self._confirmation_beep:
                self.play_beep(freq=800, duration=0.15, volume=0.3)
            if self._confirmation_tts and text:
                self.speak(text)

        threading.Thread(target=_do, daemon=True, name="audio-confirm").start()

    # ------------------------------------------------------------------
    # Device health / hot-unplug
    # ------------------------------------------------------------------

    def check_devices(self) -> dict:
        """Re-resolve device indices and report health."""
        mic_ok = False
        output_ok = False

        if self._mic_device_name:
            new_idx = self._resolve_device_index(self._mic_device_name, "input")
            if new_idx is not None:
                self._mic_device_index = new_idx
                mic_ok = True
            else:
                self._mic_device_index = None
                logger.warning(f"Mic device lost: {self._mic_device_name}")

        if self._output_device_name:
            new_idx = self._resolve_device_index(self._output_device_name, "output")
            if new_idx is not None:
                self._output_device_index = new_idx
                output_ok = True
            else:
                self._output_device_index = None
                logger.warning(f"Output device lost: {self._output_device_name}")

        return {"mic_ok": mic_ok, "output_ok": output_ok}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return full status of audio subsystem."""
        return {
            "initialized": self._initialized,
            "server_side_active": self.server_side_active,
            "mic_device_name": self._mic_device_name,
            "mic_device_index": self._mic_device_index,
            "output_device_name": self._output_device_name,
            "output_device_index": self._output_device_index,
            "vosk_model_loaded": self._vosk_model is not None,
            "vosk_model_path": self._vosk_model_path,
            "capturing": self._capturing,
            "confirmation_beep": self._confirmation_beep,
            "confirmation_tts": self._confirmation_tts,
        }

    def shutdown(self):
        """Clean up resources."""
        with self._capture_lock:
            if self._capture_stream:
                try:
                    self._capture_stream.stop()
                    self._capture_stream.close()
                except Exception:
                    pass
                self._capture_stream = None
            self._capturing = False

        with self._tts_engine_lock:
            if self._tts_engine:
                try:
                    self._tts_engine.stop()
                except Exception:
                    pass
                self._tts_engine = None

        logger.info("AudioManager shut down")
