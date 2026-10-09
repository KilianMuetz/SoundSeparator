"""apply_methods.py — alle acht Verfahren auf alle Mischsignale anwenden.

Ergebnis: ergebnisse/getrennt/<verfahren>/<mix_id>_Ns.wav und _Hs.wav
"""

from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

import verfahren_lib as vl
from daten import GETRENNT, SR, manifest, mischsignal


def trenne(aufgabe):
    e, verfahren = aufgabe
    ns, hs = vl.VERFAHREN[verfahren](mischsignal(e), SR)
    ordner = GETRENNT / verfahren
    ordner.mkdir(parents=True, exist_ok=True)
    sf.write(ordner / f"{e['mix_id']}_Ns.wav", ns, SR, subtype="FLOAT")
    sf.write(ordner / f"{e['mix_id']}_Hs.wav", hs, SR, subtype="FLOAT")


if __name__ == "__main__":
    aufgaben = [(e, v) for e in manifest for v in vl.VERFAHREN]
    with ProcessPoolExecutor() as pool:   # parallel auf allen Prozessorkernen
        list(pool.map(trenne, aufgaben))
    print(f"{len(aufgaben)} Trennungen fertig.")
