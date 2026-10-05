"""Wiener-Filter nach Scalart und Vieira Filho (1996). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "wiener"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

referenz_s = 1.0              # erste Sekunde als Referenz
glaettung = 0.98              # Anteil, der aus dem vorherigen Fenster uebernommen wird
xi_min = 10 ** (-25 / 10)     # Untergrenze von xi (-25 dB)
nperseg, noverlap = 1024, 512

# --- Laden und STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)
Y_pow = Y_mag ** 2

# --- Mittlere Leistung der Referenz ---
referenz_pow = Y_pow[:, t <= referenz_s].mean(axis=1, keepdims=True)

# --- Faktor je Fenster und Frequenz ---
Ns_mag = np.zeros_like(Y_mag)
vorher = Y_mag[:, [0]]
for k in range(Y_mag.shape[1]):
    gamma = Y_pow[:, [k]] / referenz_pow
    xi = glaettung * vorher ** 2 / referenz_pow + (1 - glaettung) * np.maximum(gamma - 1, 0)
    xi = np.maximum(xi, xi_min)
    Ns_mag[:, [k]] = xi / (1 + xi) * Y_mag[:, [k]]
    vorher = Ns_mag[:, [k]]
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
