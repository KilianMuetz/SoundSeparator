from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/synchronous_averaging/SyncAvgNs.wav"   # synchron
ausgabe_hs = BASE / "../sep/synchronous_averaging/SyncAvgHs.wav"   # asynchron

rpm_min, rpm_max = 500, 5000   # plausibler Suchbereich fuer die Drehzahl

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")

# --- Drehzahl per Autokorrelation schaetzen ---
lag_min = int(sr / (rpm_max / 60.0))           # kleinste plausible Periode
lag_max = int(sr / (rpm_min / 60.0))           # groesste plausible Periode

akf = np.correlate(y, y, mode="full")[len(y) - 1:]   # nur positive Lags

# Peak im Suchbereich; parabolische Interpolation fuer Sub-Sample-Genauigkeit
i = lag_min + np.argmax(akf[lag_min:lag_max])
a, b, c = akf[i - 1], akf[i], akf[i + 1]
delta = 0.5 * (a - c) / (a - 2 * b + c)        # Scheitel der Parabel
periode_exakt = i + delta
periode = int(round(periode_exakt))
rpm_est = 60.0 * sr / periode_exakt

# --- Signal in ganze Perioden schneiden und stapeln ---
anzahl_perioden = len(y) // periode
nutzlaenge = anzahl_perioden * periode
segmente_array = y[:nutzlaenge].reshape(anzahl_perioden, periode)

# --- Mittelung ueber alle Perioden -> synchrone Musterperiode ---
mittel_periode = segmente_array.mean(axis=0)

# --- Synchronen Anteil ueber die volle Laenge kacheln ---
ns = np.tile(mittel_periode, anzahl_perioden)
ns = np.concatenate([ns, y[nutzlaenge:]])      # Rest-Samples anhaengen
ns = ns[:len(y)]

# --- Asynchroner Rest ---
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Geschaetzte Drehzahl: {rpm_est:.1f} U/min ({sr/periode_exakt:.2f} Hz)")
print(f"Periode: {periode} Samples, {anzahl_perioden} Perioden gemittelt")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")