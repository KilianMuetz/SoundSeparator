"""sisdr.py — Trennqualitaet als SI-SDR gegen die Ground Truth (Le Roux et al. 2019)."""

import csv

import numpy as np

import verfahren_lib as vl
from daten import PROTOKOLL, ground_truth, manifest, nutzsignal


def si_sdr(schaetzung, ziel):
    """Passender Anteil der Schaetzung im Verhaeltnis zum Rest, in dB."""
    ziel, schaetzung = ziel - ziel.mean(), schaetzung - schaetzung.mean()
    anteil = np.dot(schaetzung, ziel) / np.dot(ziel, ziel) * ziel
    rest = schaetzung - anteil
    return 10 * np.log10(np.dot(anteil, anteil) / np.dot(rest, rest))


ziele = [ground_truth(e)[0] for e in manifest]


def werte(verfahren):
    return np.array([si_sdr(nutzsignal(verfahren, e), z) for e, z in zip(manifest, ziele)])


basis = werte("roh")
zeilen = [["mix_id", "nutzschall", "stoerquelle", "verfahren", "si_sdr", "si_sdri"]]
for verfahren in ["roh", *vl.VERFAHREN]:
    w = werte(verfahren)
    zeilen += [[e["mix_id"], e["nutzschall"], e["stoerquelle"], verfahren,
                round(a, 3), round(a - b, 3)] for e, a, b in zip(manifest, w, basis)]
    print(f"{verfahren:22s} SI-SDR {np.median(w):6.2f} dB   Gewinn {np.median(w - basis):+6.2f} dB")

with open(PROTOKOLL / "trennqualitaet.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
