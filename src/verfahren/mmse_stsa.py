from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.special import i0e, i1e   # exponentiell skalierte Besselfunktionen

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/mmse_stsa/mmse_stsaNs.wav"   # Nutzschall
ausgabe_hs = BASE / "../sep/mmse_stsa/mmse_stsaHs.wav"   # Hintergrundschall

rauschreferenz_ende_s = 1.0   # erste Sekunde = Rauschreferenz
alpha_dd = 0.98               # Decision-Directed-Glaettung (typ. 0.98)
xi_min = 10 ** (-25 / 10)     # a-priori-SNR-Floor (-25 dB)
nperseg = 1024
noverlap = 512

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- STFT ---
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

# --- Rauschleistung aus Referenzsegment schaetzen ---
ref_frames = t <= rauschreferenz_ende_s
noise_pow = np.mean(Y_mag[:, ref_frames] ** 2, axis=1, keepdims=True)

# --- Frame-weise MMSE-STSA-Verstaerkung ---
Y_pow = Y_mag ** 2
Ns_mag = np.zeros_like(Y_mag)
prev_amp = Y_mag[:, [0]]   # Initialisierung fuer Decision-Directed

for k in range(Y_mag.shape[1]):
    gamma = Y_pow[:, [k]] / noise_pow            # a-posteriori-SNR
    xi = alpha_dd * (prev_amp ** 2 / noise_pow) \
         + (1 - alpha_dd) * np.maximum(gamma - 1, 0)   # a-priori-SNR (DD)
    xi = np.maximum(xi, xi_min)

    v = xi / (1 + xi) * gamma
    # MMSE-STSA-Gain via exponentiell skalierter Bessel (numerisch stabil)
    gain = (np.sqrt(np.pi) / 2) * (np.sqrt(v) / gamma) \
           * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
    gain = np.minimum(gain, 1.0)                 # kein Verstaerken > 1

    Ns_mag[:, [k]] = gain * Y_mag[:, [k]]
    prev_amp = Ns_mag[:, [k]]                    # fuer naechstes Frame

# --- Hintergrundschall: komplementaerer Rest ---
Hs_mag = Y_mag - Ns_mag

# --- Rekonstruktion mit Originalphase ---
_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")