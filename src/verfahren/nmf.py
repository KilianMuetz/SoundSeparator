"""NMF nach Virtanen (2007). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "nmf"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

K = 8              # Anzahl der Grundmuster
schritte = 300     # Anzahl der Anpassungsschritte
f_grenze = 800.0   # Grundmuster mit tieferem Schwerpunkt zaehlen als Nutzschall
seed = 0
nperseg, noverlap = 1024, 512

# --- Laden und STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
V, Y_phase = np.abs(Y), np.angle(Y)

# --- Zerlegung V = W @ H mit multiplikativen Updates ---
rng = np.random.default_rng(seed)
W = rng.random((V.shape[0], K)) + 1e-6
H = rng.random((K, V.shape[1])) + 1e-6
eins = np.ones_like(V)
for _ in range(schritte):
    H *= (W.T @ (V / (W @ H + 1e-10))) / (W.T @ eins + 1e-10)
    W *= ((V / (W @ H + 1e-10)) @ H.T) / (eins @ H.T + 1e-10)

# --- Zuordnung ueber den Frequenzschwerpunkt je Grundmuster ---
schwerpunkt = (f[:, None] * W).sum(axis=0) / (W.sum(axis=0) + 1e-10)
ist_nutz = schwerpunkt <= f_grenze

# --- Anteile aufteilen ---
WH = W @ H + 1e-10
Ns_mag = (W[:, ist_nutz] @ H[ist_nutz]) / WH * V
Hs_mag = (W[:, ~ist_nutz] @ H[~ist_nutz]) / WH * V

# --- Zurueck in den Zeitbereich mit Originalphase ---
_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: {ist_nutz.sum()} von {K} Grundmustern als Nutzschall")
