"""Synchrone Mittelung nach Bechhoefer und Kingsley (2009). Prototyp fuer die Trennschaerfe."""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parents[2]
name = "synchronous_averaging"
eingabe = BASE / "data" / "observ_1.wav"
ausgabe = BASE / "explorativ" / name

rpm_min, rpm_max = 500, 5000   # plausibler Suchbereich der Drehzahl

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- Periodenlaenge per Autokorrelation schaetzen ---
lag_min = int(sr / (rpm_max / 60))
lag_max = int(sr / (rpm_min / 60))
akf = np.correlate(y, y, mode="full")[len(y) - 1:]
i = lag_min + np.argmax(akf[lag_min:lag_max])
a, b, c = akf[i - 1], akf[i], akf[i + 1]
periode = int(round(i + 0.5 * (a - c) / (a - 2 * b + c)))   # Parabel durch drei Punkte

# --- Perioden stapeln und mitteln ---
anzahl = len(y) // periode
mittel = y[:anzahl * periode].reshape(anzahl, periode).mean(axis=0)

# --- Gemittelte Periode wiederholen = Nutzschall, Rest = Stoerschall ---
ns = np.concatenate([np.tile(mittel, anzahl), y[anzahl * periode:]])
hs = y - ns

# --- Speichern ---
ausgabe.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe / f"{name}Ns.wav", ns, sr)
sf.write(ausgabe / f"{name}Hs.wav", hs, sr)
print(f"{name}: Drehzahl {60 * sr / periode:.0f} U/min, {anzahl} Perioden gemittelt")
