from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/spectral_kurtosis/spectral_kurtosisNs.wav"   # stationaer
ausgabe_hs = BASE / "../sep/spectral_kurtosis/spectral_kurtosisHs.wav"   # impulsiv

sk_schwelle = 1.0     # SK-Grenze: darueber gilt ein Band als impulsiv
mad_faktor = 3.0      # zeitlicher Ausreisser: Energie > median + k*MAD
nperseg = 1024
noverlap = 512

# --- Laden & STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag = np.abs(Y)

# --- Spectral Kurtosis je Frequenzband (ueber die Zeit) ---
# SK(f) = E[|X|^4] / E[|X|^2]^2 - 2  (die -2 gilt fuer komplexe STFT-Koeffizienten)
m2 = np.mean(Y_mag ** 2, axis=1)
m4 = np.mean(Y_mag ** 4, axis=1)
SK = m4 / (m2 ** 2 + 1e-12) - 2.0

# --- Schritt 1: impulsive Baender auswaehlen ---
impulsiv = SK > sk_schwelle

# --- Schritt 2: in diesen Baendern zeitliche Energie-Ausreisser maskieren ---
mask = np.zeros_like(Y_mag, dtype=bool)
for fi in np.where(impulsiv)[0]:
    band = Y_mag[fi, :]
    med = np.median(band)
    mad = np.median(np.abs(band - med)) + 1e-12
    mask[fi, :] = band > med + mad_faktor * mad

# --- Trennung ---
Hs = Y * mask
Ns = Y * (~mask)

# --- Rekonstruktion ---
_, ns = istft(Ns, fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs, fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"SK-Bereich: {SK.min():.1f} bis {SK.max():.1f}, {impulsiv.sum()} impulsive Baender")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")