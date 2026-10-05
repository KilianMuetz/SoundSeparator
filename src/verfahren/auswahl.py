"""Trennschaerfe der zehn Prototypen am Feldsignal observ_1.wav (Abschnitt Priorisierung)."""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parents[2]
GRENZE = 0.05   # ab diesem Rekonstruktionsfehler gilt die Rekonstruktion als verletzt

klassen = {
    "Physikalisch":   ["synchronous_averaging"],
    "Statistisch":    ["spectralsubtraction", "wiener", "mmse_stsa"],
    "Strukturell":    ["hpss", "nmf", "rpca"],
    "Modenzerlegend": ["emd", "ssa", "matching_pursuit"],
}

# --- Laden ---
x, sr = sf.read(BASE / "data" / "observ_1.wav", dtype="float64")

# --- Rekonstruktionsfehler und Unabhaengigkeit je Verfahren ---
for klasse, verfahren in klassen.items():
    print(f"\n{klasse}")
    werte = []
    for name in verfahren:
        ns, _ = sf.read(BASE / "explorativ" / name / f"{name}Ns.wav", dtype="float64")
        hs, _ = sf.read(BASE / "explorativ" / name / f"{name}Hs.wav", dtype="float64")
        n = min(len(x), len(ns), len(hs))
        eps = np.linalg.norm(x[:n] - ns[:n] - hs[:n]) / np.linalg.norm(x[:n])
        U = 1 - abs(np.corrcoef(ns[:n], hs[:n])[0, 1])
        werte.append((name, eps, U))
        print(f"  {name:22s} eps_rek = {eps:.4f}   U = {U:.4f}")
    gueltig = [w for w in werte if w[1] <= GRENZE]
    print(f"  Sieger: {max(gueltig, key=lambda w: w[2])[0]}")
