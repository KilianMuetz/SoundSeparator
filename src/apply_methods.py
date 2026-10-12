"""apply_methods.py — alle acht Verfahren auf alle Mischsignale anwenden.

Ergebnis: ergebnisse/getrennt/<verfahren>/<mix_id>_Ns.wav und _Hs.wav
          ergebnisse/protokoll/laufzeiten.csv  (Rechenzeit je Trennung)

NACHEINANDER = True misst die Rechenzeit wie auf einem Edge-Geraet, ein Kern.
NACHEINANDER = False rechnet parallel auf allen Kernen, die Zeiten sind dann
nicht aussagekraeftig.
"""

import csv
import time
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

import verfahren_lib as vl
from daten import GETRENNT, PROTOKOLL, SR, manifest, mischsignal

NACHEINANDER = True


def trenne(aufgabe):
    e, verfahren = aufgabe
    y = mischsignal(e)
    start = time.perf_counter()
    ns, hs = vl.VERFAHREN[verfahren](y, SR)   # nur die Trennung wird gemessen
    dauer = time.perf_counter() - start
    ordner = GETRENNT / verfahren
    ordner.mkdir(parents=True, exist_ok=True)
    sf.write(ordner / f"{e['mix_id']}_Ns.wav", ns, SR, subtype="FLOAT")
    sf.write(ordner / f"{e['mix_id']}_Hs.wav", hs, SR, subtype="FLOAT")
    return [e["mix_id"], verfahren, round(dauer, 4)]


if __name__ == "__main__":
    aufgaben = [(e, v) for e in manifest for v in vl.VERFAHREN]
    if NACHEINANDER:
        zeilen = [trenne(a) for a in aufgaben]
    else:
        with ProcessPoolExecutor() as pool:   # parallel auf allen Prozessorkernen
            zeilen = list(pool.map(trenne, aufgaben))

    PROTOKOLL.mkdir(parents=True, exist_ok=True)
    with open(PROTOKOLL / "laufzeiten.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["mix_id", "verfahren", "dauer_s"], *zeilen])
    print(f"{len(aufgaben)} Trennungen fertig, Laufzeiten in laufzeiten.csv.")
