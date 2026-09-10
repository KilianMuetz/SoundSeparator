"""
apply_methods.py — Wendet alle ausgewaehlten Trennverfahren auf alle Mischsignale an.

Vollstaendige Kombination: 228 Mischsignale x 8 Verfahren = 1824 Trennungen.
Ergebnis je Verfahren und Mischsignal:
  ergebnisse/getrennt/<verfahren>/<mix_id>_Ns.wav und ..._Hs.wav

Die Mischsignale liegen mit 44,1 kHz vor und werden auf 16 kHz dezimiert,
der Abtastrate, mit der SIPREMA auf dem Edge Device arbeitet.

Aufruf:  python src/apply_methods.py
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import verfahren_lib as vl

# --- Pfade ---
BASE = Path(__file__).parent.parent
mix_dir = BASE / "data" / "mixed"
getrennt_dir = BASE / "ergebnisse" / "getrennt"
protokoll = BASE / "ergebnisse" / "protokoll" / "verfahrensanwendung.csv"

# --- Parameter ---
SR_ZIEL = 16000     # Abtastrate des SIPREMA-Edge-Devices
SR_QUELLE = 44100
UEBERSPRINGEN = True   # bereits protokollierte Ergebnisse nicht neu berechnen


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


# --- Mischsignale laden ---
with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"{len(manifest)} Mischsignale x {len(vl.VERFAHREN)} Verfahren "
      f"= {len(manifest) * len(vl.VERFAHREN)} Trennungen\n")

# --- Bereits protokollierte Trennungen einlesen (Wiederaufsetzen nach Abbruch) ---
fertig = set()
if protokoll.exists():
    with open(protokoll, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fertig.add((r["mix_id"], r["verfahren"]))
    print(f"{len(fertig)} Trennungen bereits protokolliert, werden uebersprungen.\n")

zeilen = []
t_start = time.time()

for i, eintrag in enumerate(manifest, 1):
    mix_id = eintrag["mix_id"]
    y_roh, sr_roh = sf.read(mix_dir / eintrag["file"], dtype="float64")
    assert sr_roh == SR_QUELLE, f"{mix_id}: Abtastrate {sr_roh} != {SR_QUELLE}"

    y = resample_poly(y_roh, SR_ZIEL // 100, SR_QUELLE // 100)
    energie_y = float(np.sum(y ** 2))

    for verfahren, trenne in vl.VERFAHREN.items():
        ziel_ns = getrennt_dir / verfahren / f"{mix_id}_Ns.wav"
        ziel_hs = getrennt_dir / verfahren / f"{mix_id}_Hs.wav"

        if UEBERSPRINGEN and (mix_id, verfahren) in fertig \
                and ziel_ns.exists() and ziel_hs.exists():
            continue

        t0 = time.time()
        ns, hs = trenne(y, SR_ZIEL)
        dauer = time.time() - t0

        ziel_ns.parent.mkdir(parents=True, exist_ok=True)
        sf.write(ziel_ns, ns, SR_ZIEL, subtype="FLOAT")
        sf.write(ziel_hs, hs, SR_ZIEL, subtype="FLOAT")

        # Kontrollgroessen: Rekonstruktionsfehler und Energieanteil des Nutzschalls
        eps_rek = rms(y - (ns + hs)) / rms(y)
        anteil_ns = float(np.sum(ns ** 2)) / (energie_y + 1e-12)

        zeilen.append({
            "mix_id": mix_id,
            "nutzschall": eintrag["nutzschall"],
            "stoerquelle": eintrag["stoerquelle"],
            "verfahren": verfahren,
            "dauer_s": round(dauer, 3),
            "eps_rek": round(eps_rek, 6),
            "anteil_ns": round(anteil_ns, 4),
        })

    print(f"[{i:3d}/{len(manifest)}] {mix_id} {eintrag['nutzschall']:12s} "
          f"{eintrag['stoerquelle']:16s} "
          f"({time.time() - t_start:.0f}s)")

# --- Protokoll schreiben (Grundlage fuer die Laufzeitbetrachtung in Kapitel 6) ---
protokoll.parent.mkdir(parents=True, exist_ok=True)
if zeilen:
    neu = not protokoll.exists()
    with open(protokoll, "a", newline="", encoding="utf-8") as f:
        schreiber = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()))
        if neu:
            schreiber.writeheader()
        schreiber.writerows(zeilen)

print(f"\n{len(zeilen)} Trennungen berechnet in {time.time() - t_start:.0f}s.")
print(f"Protokoll: {protokoll}")
