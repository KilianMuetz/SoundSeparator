from pathlib import Path

import numpy as np
import soundfile as sf
import pywt

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/wavelet/WaveletNs.wav"   # stationaer
ausgabe_hs = BASE / "../sep/wavelet/WaveletHs.wav"   # transient

wavelet = "db8"   # Daubechies-8 (guter Kompromiss Zeit/Frequenz)
level = 6         # Zerlegungstiefe
k = 2.5           # Schwellfaktor (klein = mehr wird als transient erkannt)

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Diskrete Wavelet-Zerlegung ---
coeffs = pywt.wavedec(y, wavelet, level=level)

# --- Transienten-Isolation je Detailskala ---
coeffs_ns = [coeffs[0].copy()]                 # Approximation -> stationaer
coeffs_hs = [np.zeros_like(coeffs[0])]
for c in coeffs[1:]:
    sigma = np.median(np.abs(c)) / 0.6745 + 1e-12   # robuste Streuung (MAD)
    thr = k * sigma
    transient = np.abs(c) > thr                      # grosse Ausreisser
    coeffs_hs.append(np.where(transient, c, 0.0))    # Stoerschall = Ausreisser
    coeffs_ns.append(np.where(transient, 0.0, c))    # Nutzschall = Rest

# --- Rekonstruktion ---
ns = pywt.waverec(coeffs_ns, wavelet)[:N]
hs = pywt.waverec(coeffs_hs, wavelet)[:N]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Wavelet {wavelet}, {level} Ebenen, k = {k}")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")