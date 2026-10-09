"""auswertung.py — Tabellen fuer Kapitel 6 aus den Protokollen.

Erkennung je Anomalie, Stoerart und Schweregrad sowie Trennqualitaet je Stoerart.
"""

import csv

import numpy as np
from sklearn.metrics import roc_auc_score

from daten import PROTOKOLL, manifest

HINTERGRUND = {"radio", "verkehr", "akku_konstant", "akku_dynamisch"}
ZUSTAENDE = ["normal", "a1_leicht", "a1_stark", "a2_leicht", "a2_deutlich",
             "a3_leicht", "a3_deutlich"]
info = {e["mix_id"]: e for e in manifest}


def lesen(name):
    with open(PROTOKOLL / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stoerart(e):
    return "hintergrund" if e["stoerquelle"] in HINTERGRUND else "fremd"


def auc(paare, auswahl):
    """AUC nur ueber die ausgewaehlten Segmente."""
    paare = [(e, w) for e, w in paare if auswahl(e)]
    return roc_auc_score([e["nutzschall"] != "normal" for e, _ in paare], [w for _, w in paare])


print("Erkennung: AUC je Anomalie und Stoerart, mittlerer Score je Zustand")
print(f"{'':22s}" + "".join(f"{k:>8s}" for k in ["A1", "A2", "A3", "Fremd", "Hint."])
      + "".join(f"{z:>12s}" for z in ZUSTAENDE))
scores = lesen("erkennung.csv")
for verfahren in dict.fromkeys(z["verfahren"] for z in scores):
    paare = [(info[z["mix_id"]], float(z["score"])) for z in scores if z["verfahren"] == verfahren]
    werte = [auc(paare, lambda e: e["nutzschall"] == "normal" or e["nutzschall"].startswith(a))
             for a in ["a1", "a2", "a3"]]
    werte += [auc(paare, lambda e: stoerart(e) == g) for g in ["fremd", "hintergrund"]]
    mittel = [np.mean([w for e, w in paare if e["nutzschall"] == z]) for z in ZUSTAENDE]
    print(f"{verfahren:22s}" + "".join(f"{w:8.3f}" for w in werte)
          + "".join(f"{m:12.3f}" for m in mittel))

print("\nTrennqualitaet: Median des Gewinns in dB je Stoerart")
zeilen = lesen("trennqualitaet.csv")
for verfahren in dict.fromkeys(z["verfahren"] for z in zeilen):
    gewinn = {g: np.median([float(z["si_sdri"]) for z in zeilen if z["verfahren"] == verfahren
                            and stoerart(info[z["mix_id"]]) == g]) for g in ["fremd", "hintergrund"]}
    print(f"{verfahren:22s} Fremd {gewinn['fremd']:+6.2f}   Hintergrund {gewinn['hintergrund']:+6.2f}")

