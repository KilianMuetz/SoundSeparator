from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/combfilter/CombNs.wav"   # Harmonische = Nutzschall
ausgabe_hs = BASE / "../sep/combfilter/CombHs.wav"   # Rest        = Stoerschall

f_r = 25.0            # Rotationsgrundfrequenz in Hz (= rpm/60)
bandbreite = 4.0      # halbe Zahnbreite je Harmonische in Hz
nperseg = 1024
noverlap = 512

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- STFT ---
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)

# --- Kamm-Maske: Baender um jede Harmonische bis zur Nyquist-Frequenz ---
anzahl_harmonische = int((sr / 2) / f_r)          # nur bis Nyquist -> kein Alias
mask = np.zeros(len(f), dtype=bool)
for n in range(1, anzahl_harmonische + 1):
    harmonische = n * f_r
    mask |= np.abs(f - harmonische) <= bandbreite

# TODO: bei sehr niedrigem f_r ueberlappen benachbarte Zaehne ->
#       Bandbreite adaptiv < f_r/2 begrenzen

# --- Trennung ueber die Frequenzmaske ---
Ns = Y * mask[:, None]        # nur Kamm-Baender
Hs = Y * (~mask)[:, None]     # komplementaerer Rest

# --- Rekonstruktion ---
_, ns = istft(Ns, fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs, fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"{anzahl_harmonische} Harmonische gefiltert (f_r = {f_r} Hz)")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")