from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.ndimage import median_filter

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_h = BASE / "../sep/hpss/HpssNs.wav"   # harmonisch = Maschinengeraeusch
ausgabe_p = BASE / "../sep/hpss/HpssSs.wav"   # percussiv  = Schlaege

kernel_zeit = 51      # Medianfilter-Laenge entlang Zeit  (holt Harmonisches)
kernel_freq = 17      # Medianfilter-Laenge entlang Frequenz (holt Percussives)
power = 2.0           # Maskenschaerfe (1 = weich, >1 = harter Wiener)
nperseg = 1024
noverlap = 512

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- STFT ---
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

# --- Zwei Referenzen per Medianfilter ---
H_ref = median_filter(Y_mag, size=(1, kernel_zeit))   # entlang Zeit  -> harmonisch
P_ref = median_filter(Y_mag, size=(kernel_freq, 1))   # entlang Freq  -> percussiv

# --- Weiche Wiener-artige Masken ---
eps = 1e-12
H_ref_p, P_ref_p = H_ref ** power, P_ref ** power
mask_h = H_ref_p / (H_ref_p + P_ref_p + eps)
mask_p = P_ref_p / (H_ref_p + P_ref_p + eps)

H_mag = Y_mag * mask_h
P_mag = Y_mag * mask_p

# --- Rekonstruktion mit Originalphase ---
_, h = istft(H_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, p = istft(P_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
h, p = h[:len(y)], p[:len(y)]

# --- Speichern ---
ausgabe_h.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_h, h, sr)
sf.write(ausgabe_p, p, sr)
print(f"Fertig: {ausgabe_h}")
print(f"Fertig: {ausgabe_p}")