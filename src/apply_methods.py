"""apply_methods.py — alle acht Verfahren auf alle Mischsignale anwenden.

Ergebnis: ergebnisse/getrennt/<verfahren>/<mix_id>_Ns.wav und _Hs.wav,
dazu die Rechenzeit je Trennung in verfahrensanwendung.csv.
"""

import csv
import time
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

import verfahren_lib as vl
from daten import GETRENNT, PROTOKOLL, SR, manifest, mischsignal


def trenne(aufgabe):
    e, verfahren = aufgabe
    y = mischsignal(e)
    start = time.time()
    ns, hs = vl.VERFAHREN[verfahren](y, SR)
    dauer = time.time() - start

    ordner = GETRENNT / verfahren
    ordner.mkdir(parents=True, exist_ok=True)
    sf.write(ordner / f"{e['mix_id']}_Ns.wav", ns, SR, subtype="FLOAT")
    sf.write(ordner / f"{e['mix_id']}_Hs.wav", hs, SR, subtype="FLOAT")
    return [e["mix_id"], e["nutzschall"], e["stoerquelle"], verfahren, round(dauer, 3)]


if __name__ == "__main__":
    aufgaben = [(e, v) for e in manifest for v in vl.VERFAHREN]
    with ProcessPoolExecutor() as pool:   # parallel auf allen Prozessorkernen
        zeilen = list(pool.map(trenne, aufgaben))

    with open(PROTOKOLL / "verfahrensanwendung.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [["mix_id", "nutzschall", "stoerquelle", "verfahren", "dauer_s"]] + zeilen)
    print(f"{len(zeilen)} Trennungen fertig.")
