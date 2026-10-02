from pathlib import Path
import numpy as np
import soundfile as sf

BASE = Path(__file__).parents[2]
GRENZE = 0.05   # ab diesem Rekonstruktionsfehler gilt die Rekonstruktion als verletzt

klassen = {
    "Physikalisch":   ["synchronous_averaging"],
    "Statistisch":    ["spectralsubtraction", "wiener", "mmse_stsa"],
    "Strukturell":    ["hpss", "nmf", "rpca"],
    "Modenzerlegend": ["emd", "vmd", "ssa", "matching_pursuit"],
}

x, sr = sf.read(BASE / "data" / "observ_1.wav")

for klasse, verfahren in klassen.items():
    print(f"\n{klasse}")
    werte = []
    for name in verfahren:
        o, _ = sf.read(BASE / "explorativ" / name / f"{name}Ns.wav")
        s, _ = sf.read(BASE / "explorativ" / name / f"{name}Hs.wav")
        n = min(len(x), len(o), len(s))
        eps = np.linalg.norm(x[:n] - o[:n] - s[:n]) / np.linalg.norm(x[:n])
        r = abs(np.corrcoef(o[:n], s[:n])[0, 1])
        werte.append((name, eps, r))
        print(f"  {name:22s} eps = {eps:.4f}   |r| = {r:.4f}")
    gueltig = [w for w in werte if w[1] <= GRENZE]
    print(f"  Sieger: {min(gueltig, key=lambda w: w[2])[0]}")
