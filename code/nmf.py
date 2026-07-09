from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/nmf/NmfNs.wav"
ausgabe_hs = BASE / "../sep/nmf/NmfHs.wav"

f_grenze = 800.0  # Obergrenze des Maschinenbands in Hz (Zuordnungskriterium)
K = 8             # Anzahl der Komponenten
n_iter = 300      # Iterationen der multiplikativen Updates
seed = 0          # Zufalls-Init (Reproduzierbarkeit)
nperseg = 1024
noverlap = 512

# --- Laden & STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
V, Yph = np.abs(Y), np.angle(Y)

# --- NMF mit KL-Divergenz (multiplikative Updates) ---
rng = np.random.default_rng(seed)
m, n = V.shape
W = rng.random((m, K)) + 1e-6
H = rng.random((K, n)) + 1e-6
eps = 1e-10
ones = np.ones((m, n))

for it in range(n_iter):
    WH = W @ H + eps
    H *= (W.T @ (V / WH)) / (W.T @ ones + eps)
    WH = W @ H + eps
    W *= ((V / WH) @ H.T) / (ones @ H.T + eps)

# --- Zuordnung ueber den spektralen Schwerpunkt je Komponente ---
centroid = (f[:, None] * W).sum(axis=0) / (W.sum(axis=0) + eps)
ist_nutz = centroid <= f_grenze

# --- Wiener-artige Rekonstruktion je Kanal ---
WH_ges = W @ H + eps
V_ns = (W[:, ist_nutz] @ H[ist_nutz]) / WH_ges * V
V_hs = (W[:, ~ist_nutz] @ H[~ist_nutz]) / WH_ges * V

_, ns = istft(V_ns * np.exp(1j * Yph), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(V_hs * np.exp(1j * Yph), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Centroids (Hz): {np.round(np.sort(centroid), 0)}")
print(f"Nutzschall-Komponenten: {np.where(ist_nutz)[0].tolist()}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")