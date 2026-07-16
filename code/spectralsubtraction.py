from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/spectralsubtraction/spectralsubtractionNs.wav"
ausgabe_hs = BASE / "../sep/spectralsubtraction/spectralsubtractionHs.wav"

rauschreferenz_ende_s = 1.0   # erste Sekunde = Rauschreferenz
alpha = 2.0                   # Oversubtraction-Faktor
beta = 0.02                   # Spectral Floor
nperseg = 1024
noverlap = 512

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- STFT ---
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

# --- Rauschspektrum aus Referenzsegment schaetzen ---
ref_frames = t <= rauschreferenz_ende_s
noise_mag = np.mean(Y_mag[:, ref_frames], axis=1, keepdims=True)

# --- Nutzschall = das stationaere Maschinengeraeusch (aus Referenz geschaetzt) ---
Ns_mag = Y_mag - np.maximum(Y_mag - alpha * noise_mag, beta * Y_mag)

# --- Hintergrund/Stoerschall = der nicht-stationaere Rest ---
Hs_mag = Y_mag - Ns_mag

# --- Rekonstruktion mit Originalphase ---
_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")