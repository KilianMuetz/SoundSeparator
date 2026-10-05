"""SSA nach Hassani (2007). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "ssa"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

L = 300          # Fensterlaenge
r = 5            # so viele staerkste Komponenten zaehlen als Nutzschall
block = 8000     # Blocklaenge in Abtastwerten

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- Blockweise zerlegen ---
ns = y.copy()                                   # zu kurzer Restblock bleibt unveraendert
for start in range(0, len(y), block):
    seg = y[start:start + block]
    if len(seg) < L + 10:
        continue
    K = len(seg) - L + 1
    X = np.column_stack([seg[i:i + L] for i in range(K)])   # verschobene Ausschnitte
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_r = (U[:, :r] * S[:r]) @ Vt[:r]                         # staerkste r Komponenten
    # zurueck in ein Zeitsignal: Mittelwert entlang der Gegendiagonalen
    summe, anzahl = np.zeros(len(seg)), np.zeros(len(seg))
    for i in range(L):
        summe[i:i + K] += X_r[i]
        anzahl[i:i + K] += 1
    ns[start:start + len(seg)] = summe / anzahl
hs = y - ns

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: fertig")
