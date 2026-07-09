from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/cyclostationary/CycloNs.wav"   # zyklostationaer
ausgabe_hs = BASE / "../sep/cyclostationary/CycloHs.wav"   # residual

f_r = 25.0            # Zyklusfrequenz (Rotationsgrundfrequenz) in Hz
n_harmonische = 5     # Anzahl beruecksichtigter Zyklus-Harmonischer
nperseg = 1024
noverlap = 512

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- STFT ---
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

# --- Huellkurve je Frequenzband (mittelwertbereinigt) ---
env = Y_mag - Y_mag.mean(axis=1, keepdims=True)   # DC pro Band entfernen
dt = t[1] - t[0]                                  # Frame-Abstand in s

# --- Zyklische Modulationsstaerke bei alpha = n * f_r messen ---
# Projektion der Bandhuellkurve auf komplexe Schwingungen exp(-j 2pi alpha t):
# grosse Betragswerte -> Band moduliert drehzahlsynchron -> zyklostationaer
Px = np.zeros(len(f))
for n in range(1, n_harmonische + 1):
    alpha = n * f_r
    schwingung = np.exp(-1j * 2 * np.pi * alpha * t)   # ueber Frame-Zeiten t
    Px += np.abs(env @ schwingung)                     # je Band aufsummiert

# --- Maske normieren auf [0, 1] ---
Px_norm = Px / (Px.max() + 1e-12)
mask = Px_norm[:, None]        # als Zeit-invariante Bandgewichtung

# --- Trennung ---
Ns = Y * mask
Hs = Y * (1 - mask)

# --- Rekonstruktion ---
_, ns = istft(Ns, fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs, fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Zyklusfrequenz alpha = {f_r} Hz, {n_harmonische} Harmonische")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")