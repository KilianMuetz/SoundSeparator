"""Trennschaerfe der zehn Prototypen am Feldsignal observ_1.wav."""

from pathlib import Path

import numpy as np
import soundfile as sf

from verfahren_lib import PROTOTYPEN

BASE = Path(__file__).parents[1]
y, sr = sf.read(BASE / "data" / "observ_1.wav", dtype="float64")

for name, trenne in PROTOTYPEN.items():
    ns, hs = trenne(y, sr)
    eps_rek = np.linalg.norm(y - (ns + hs)) / np.linalg.norm(y)   # Rekonstruktionsfehler
    U = 1 - abs(np.corrcoef(ns, hs)[0, 1])                        # Unabhaengigkeit
    print(f"{name:22s} eps_rek = {eps_rek:.4f}   U = {U:.3f}")
