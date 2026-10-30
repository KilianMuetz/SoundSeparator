"""
laufzeit_box.py — Rechenzeit der Trennstufe auf der SIPREMA-Box.

Laeuft im Backend-Container der Box. Misst je 5-s-Segment:
  trennung  = SpectralSubtraction, Code identisch mit feature/trennstufe
  merkmale  = AudioFeatures.from_audio_bytes, wie die Box es fuer jedes Segment tut
"""

import time
from datetime import datetime

import numpy as np
import scipy.signal as signal

from src.application.domain.audio_features import AudioFeatures

SR = 16000
N = 50   # Wiederholungen


def spectral_subtraction(audio_data, sample_rate, percentile=20.0, floor=0.04, frame_length=1024):
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
    hop = frame_length // 2
    _, _, spectrum = signal.stft(samples, fs=sample_rate, nperseg=frame_length, noverlap=hop)
    magnitude = np.abs(spectrum)
    stationary = np.percentile(magnitude, percentile, axis=1, keepdims=True)
    machine = np.minimum(stationary, (1.0 - floor) * magnitude)
    _, separated = signal.istft(machine * np.exp(1j * np.angle(spectrum)),
                                fs=sample_rate, nperseg=frame_length, noverlap=hop)
    separated = np.clip(separated[: len(samples)] * 32767.0, -32768, 32767)
    return separated.astype(np.int16).tobytes()


def messen(funktion):
    """Median der Rechenzeit in ms, erster Aufruf nicht gezaehlt."""
    funktion()
    zeiten = []
    for _ in range(N):
        start = time.perf_counter()
        funktion()
        zeiten.append(time.perf_counter() - start)
    return 1000 * np.median(zeiten)


pcm = (np.random.default_rng(0).standard_normal(5 * SR) * 3000).astype(np.int16).tobytes()

trennung = messen(lambda: spectral_subtraction(pcm, SR))
merkmale = messen(lambda: AudioFeatures.from_audio_bytes(pcm, SR, datetime.now()))

print(f"Trennung  {trennung:7.1f} ms je 5-s-Segment")
print(f"Merkmale  {merkmale:7.1f} ms je 5-s-Segment")
print(f"Anteil    {trennung / merkmale * 100:7.1f} % der Merkmalsberechnung")
print(f"Budget    {trennung / 5000 * 100:7.2f} % von 5 s")
