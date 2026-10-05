"""Spektralsubtraktion nach Boll (1979). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "spectralsubtraction"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

referenz_s = 1.0   # erste Sekunde als Referenz
alpha = 2.0        # wie stark abgezogen wird
beta = 0.02        # Untergrenze, damit nichts ganz verschwindet
nperseg, noverlap = 1024, 512

# --- Laden und STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

# --- Mittleres Spektrum der Referenz ---
referenz = Y_mag[:, t <= referenz_s].mean(axis=1, keepdims=True)

# --- Subtraktion ---
Ns_mag = Y_mag - np.maximum(Y_mag - alpha * referenz, beta * Y_mag)
Hs_mag = Y_mag - Ns_mag

# --- Zurueck in den Zeitbereich mit Originalphase ---
_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: fertig")
