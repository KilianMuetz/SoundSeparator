"""HPSS nach FitzGerald (2010). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "hpss"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

laenge_zeit = 51   # Medianfilter entlang der Zeit, holt die Toene
laenge_freq = 17   # Medianfilter entlang der Frequenz, holt die Schlaege
p = 2.0            # Schaerfe der Maske
nperseg, noverlap = 1024, 512

# --- Laden und STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

# --- Waagerechte und senkrechte Anteile per Medianfilter ---
H = median_filter(Y_mag, size=(1, laenge_zeit)) ** p
P = median_filter(Y_mag, size=(laenge_freq, 1)) ** p

# --- Wer staerker ist, bekommt den groesseren Anteil ---
Ns_mag = Y_mag * H / (H + P + 1e-12)
Hs_mag = Y_mag * P / (H + P + 1e-12)

# --- Zurueck in den Zeitbereich mit Originalphase ---
_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: fertig")
