"""Matching Pursuit nach Mallat und Zhang (1993). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "matching_pursuit"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

bausteine = 100    # so viele Bausteine je Block
f_grenze = 800.0   # Bausteine mit tieferer Frequenz zaehlen als Nutzschall
block = 4096       # Blocklaenge in Abtastwerten
versatz = 2048     # halbe Blocklaenge, 50 % Ueberlappung

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
fenster = np.hanning(block)
freqs = np.fft.rfftfreq(block, 1 / sr)

# --- Blockweise zerlegen und ueberlappend zusammensetzen ---
ns, gewicht = np.zeros_like(y), np.zeros_like(y)
for start in range(0, len(y) - block + 1, versatz):
    rest = y[start:start + block].copy()
    ns_block = np.zeros(block)
    for _ in range(bausteine):
        R = np.fft.rfft(rest)
        idx = np.argmax(np.abs(R))                 # staerkster Baustein
        spek = np.zeros_like(R)
        spek[idx] = R[idx]
        baustein = np.fft.irfft(spek, n=block)
        if freqs[idx] <= f_grenze:
            ns_block += baustein
        rest -= baustein
    ns[start:start + block] += ns_block * fenster
    gewicht[start:start + block] += fenster
ns = ns / np.where(gewicht < 1e-8, 1.0, gewicht)
hs = y - ns

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: fertig")
