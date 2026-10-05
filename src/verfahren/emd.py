"""EMD nach Huang et al. (1998). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf
from PyEMD import EMD

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "emd"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

max_moden = 10
f_grenze = 800.0   # Moden mit tieferem Schwerpunkt zaehlen als Nutzschall

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- Zerlegung in Moden ---
moden = EMD().emd(y, max_imf=max_moden)

# --- Zuordnung ueber den Frequenzschwerpunkt je Mode ---
freqs = np.fft.rfftfreq(len(y), 1 / sr)
ns, hs = np.zeros_like(y), np.zeros_like(y)
for mode in moden:
    spek = np.abs(np.fft.rfft(mode))
    if np.sum(freqs * spek) / (np.sum(spek) + 1e-12) <= f_grenze:
        ns += mode
    else:
        hs += mode

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: {len(moden)} Moden")
