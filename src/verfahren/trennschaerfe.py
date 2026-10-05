"""Trennschaerfe der zehn Prototypen am Feldsignal observ_1.wav."""

from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).parents[2]
verfahren = ["synchronous_averaging", "spectralsubtraction", "wiener", "mmse_stsa",
             "hpss", "nmf", "rpca", "emd", "ssa", "matching_pursuit"]

y, sr = sf.read(BASE / "data" / "observ_1.wav")

for name in verfahren:
    ns, _ = sf.read(BASE / "explorativ" / name / f"{name}Ns.wav")
    hs, _ = sf.read(BASE / "explorativ" / name / f"{name}Hs.wav")
    n = min(len(y), len(ns), len(hs))
    y_, ns, hs = y[:n], ns[:n], hs[:n]

    eps_rek = np.linalg.norm(y_ - (ns + hs)) / np.linalg.norm(y_)   # Rekonstruktionsfehler
    U = 1 - abs(np.corrcoef(ns, hs)[0, 1])                          # Unabhaengigkeit

    print(f"{name:22s} eps_rek = {eps_rek:.4f}   U = {U:.3f}")
