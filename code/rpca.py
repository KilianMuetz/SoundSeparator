from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/rpca/RpcaNs.wav"   # low-rank = Nutzschall
ausgabe_hs = BASE / "../sep/rpca/RpcaHs.wav"   # sparse   = Stoerschall

lam = None        # Sparse-Gewicht (None -> 1/sqrt(max(m,n)) nach Candes)
tol = 1e-7        # Konvergenzschwelle
max_iter = 500
nperseg = 1024
noverlap = 512

# --- Laden & STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
M, Yph = np.abs(Y), np.angle(Y)

# --- RPCA via Inexact ALM ---
m, n = M.shape
if lam is None:
    lam = 1.0 / np.sqrt(max(m, n))
norm2 = np.linalg.norm(M, 2)                    # groesster Singulaerwert
normF = np.linalg.norm(M, "fro")
mu = 1.25 / norm2
mu_bar = mu * 1e7
rho = 1.5

L = np.zeros_like(M)
S = np.zeros_like(M)
Yd = M / max(norm2, np.linalg.norm(M, np.inf) / lam)   # duale Variable

for it in range(max_iter):
    # L-Update: Singulaerwert-Schwellung von (M - S + Yd/mu)
    U, sig, Vt = np.linalg.svd(M - S + Yd / mu, full_matrices=False)
    sig_t = np.maximum(sig - 1 / mu, 0)
    L = (U * sig_t) @ Vt
    # S-Update: Soft-Thresholding von (M - L + Yd/mu)
    Tm = M - L + Yd / mu
    S = np.sign(Tm) * np.maximum(np.abs(Tm) - lam / mu, 0)
    # duales Update
    Z = M - L - S
    Yd = Yd + mu * Z
    mu = min(mu * rho, mu_bar)
    if np.linalg.norm(Z, "fro") / normF < tol:
        break

# --- Rekonstruktion mit Originalphase ---
_, ns = istft(L * np.exp(1j * Yph), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(S * np.exp(1j * Yph), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Konvergiert nach {it} Iterationen")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")