"""RPCA nach Candes et al. (2011). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "rpca"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

max_schritte = 500
toleranz = 1e-7
nperseg, noverlap = 1024, 512

# --- Laden und STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
M, Y_phase = np.abs(Y), np.angle(Y)

# --- Startwerte ---
lam = 1 / np.sqrt(max(M.shape))   # Gewicht des seltenen Teils nach Candes et al.
norm2 = np.linalg.norm(M, 2)
mu, mu_max, rho = 1.25 / norm2, 1.25 / norm2 * 1e7, 1.5
L, S = np.zeros_like(M), np.zeros_like(M)
Yd = M / max(norm2, np.linalg.norm(M, np.inf) / lam)

# --- Zerlegung M = L + S ---
for _ in range(max_schritte):
    U, sig, Vt = np.linalg.svd(M - S + Yd / mu, full_matrices=False)
    L = (U * np.maximum(sig - 1 / mu, 0)) @ Vt                          # wiederkehrend
    T = M - L + Yd / mu
    S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0)                # selten
    Z = M - L - S
    Yd = Yd + mu * Z
    mu = min(mu * rho, mu_max)
    if np.linalg.norm(Z, "fro") / np.linalg.norm(M, "fro") < toleranz:
        break

# --- Zurueck in den Zeitbereich mit Originalphase ---
_, ns = istft(L * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(S * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: fertig")
