"""
apply_methods.py — Wendet alle ausgewaehlten Trennverfahren auf alle Mischsignale an.

Vollstaendige Kombination: 228 Mischsignale x 8 Verfahren = 1824 Trennungen.
Ergebnis je Verfahren und Mischsignal:
  ergebnisse/getrennt/<verfahren>/<mix_id>_Ns.wav und ..._Hs.wav

Die Mischsignale liegen mit 44,1 kHz vor und werden auf 16 kHz dezimiert,
der Abtastrate, mit der SIPREMA auf dem Edge Device arbeitet.

Die 1824 (Mischsignal, Verfahren)-Paare sind voneinander unabhaengig und
werden daher ueber einen Prozesspool parallelisiert (Standard: alle Kerne).
Das Protokoll wird laufend geschrieben (nicht erst am Ende), damit ein
Abbruch mitten im Lauf keine bereits berechneten Trennungen kostet.

Aendert sich verfahren_lib.py, wird das am Hash im Protokoll erkannt: alte
Zeilen mit abweichendem Hash gelten nicht mehr als "fertig" und werden neu
berechnet, statt stillschweigend veraltete Ergebnisse zu behalten.

Aufruf:  python src/apply_methods.py [n_worker]
"""

import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
UEBERSPRINGEN = True   # bereits protokollierte Ergebnisse (gleicher Lib-Hash) nicht neu berechnen
FELDER = ["mix_id", "nutzschall", "stoerquelle", "verfahren",
          "dauer_s", "eps_rek", "anteil_ns", "lib_hash"]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def lib_hash():
    """Kurzer Hash von verfahren_lib.py -- Aenderungen daran entwerten alte Protokollzeilen."""
    return hashlib.sha256(Path(vl.__file__).read_bytes()).hexdigest()[:12]


def bereits_erledigt(protokoll_pfad, aktueller_hash):
    """Liest fruehere Laeufe ein; Zeilen mit altem Lib-Hash zaehlen nicht als fertig.

    Ein Protokoll aus der Zeit vor diesem Feld (kein "lib_hash") wird als
    veraltet behandelt: es wird als .bak gesichert, alle betroffenen Paare
    werden neu gerechnet, statt mit unbekanntem Hash stillschweigend als
    gueltig zu gelten.
    """
    if not protokoll_pfad.exists():
        return set()

    with open(protokoll_pfad, newline="", encoding="utf-8") as f:
        zeilen = list(csv.DictReader(f))

    if zeilen and "lib_hash" not in zeilen[0]:
        sicherung = protokoll_pfad.with_suffix(".csv.bak")
        protokoll_pfad.rename(sicherung)
        print(f"Protokoll ohne Versions-Hash gefunden, gesichert nach {sicherung}. "
              f"Alle Trennungen werden neu berechnet.\n")
        return set()

    fertig = {(r["mix_id"], r["verfahren"]) for r in zeilen if r.get("lib_hash") == aktueller_hash}
    veraltet = len(zeilen) - len(fertig)
    print(f"{len(fertig)} Trennungen bereits protokolliert (aktueller Stand), werden uebersprungen.")
    if veraltet:
        print(f"{veraltet} Zeilen mit abweichendem Lib-Hash gefunden, werden neu berechnet.")
    print()
    return fertig


def _verarbeite(mix_id, verfahren, y):
    """Laeuft im Worker-Prozess: eine Trennung, ein (mix, verfahren)-Paar."""
    trenne = vl.VERFAHREN[verfahren]
    t0 = time.time()
    ns, hs = trenne(y, SR_ZIEL)
    dauer = time.time() - t0

    energie_y = float(np.sum(y ** 2))
    eps_rek = rms(y - (ns + hs)) / rms(y)
    anteil_ns = float(np.sum(ns ** 2)) / (energie_y + 1e-12)

    return mix_id, verfahren, ns, hs, dauer, eps_rek, anteil_ns


def main(n_worker=None):
    with open(mix_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    n_trennungen = len(manifest) * len(vl.VERFAHREN)
    print(f"{len(manifest)} Mischsignale x {len(vl.VERFAHREN)} Verfahren "
          f"= {n_trennungen} Trennungen\n")

    aktueller_hash = lib_hash()
    fertig = bereits_erledigt(protokoll, aktueller_hash)

    # --- Mischsignale einmal je Mix laden/dezimieren, offene Aufgaben sammeln ---
    signale, eintraege, aufgaben = {}, {}, []
    for eintrag in manifest:
        mix_id = eintrag["mix_id"]
        eintraege[mix_id] = eintrag

        offene = [v for v in vl.VERFAHREN
                  if not (UEBERSPRINGEN and (mix_id, v) in fertig
                          and (getrennt_dir / v / f"{mix_id}_Ns.wav").exists()
                          and (getrennt_dir / v / f"{mix_id}_Hs.wav").exists())]
        if not offene:
            continue

        y_roh, sr_roh = sf.read(mix_dir / eintrag["file"], dtype="float64")
        assert sr_roh == SR_QUELLE, f"{mix_id}: Abtastrate {sr_roh} != {SR_QUELLE}"
        signale[mix_id] = resample_poly(y_roh, SR_ZIEL // 100, SR_QUELLE // 100)
        aufgaben.extend((mix_id, v) for v in offene)

    if not aufgaben:
        print("Nichts zu tun -- alle Trennungen bereits auf aktuellem Stand.")
        return

    print(f"{len(aufgaben)} offene Trennungen, verteilt auf "
          f"{n_worker or 'alle verfuegbaren'} Prozesse.\n")

    protokoll.parent.mkdir(parents=True, exist_ok=True)
    neu = not protokoll.exists()
    t_start = time.time()
    erledigt = 0

    with open(protokoll, "a", newline="", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=n_worker) as pool:
        schreiber = csv.DictWriter(f, fieldnames=FELDER)
        if neu:
            schreiber.writeheader()

        futures = [pool.submit(_verarbeite, mix_id, verfahren, signale[mix_id])
                   for mix_id, verfahren in aufgaben]

        for future in as_completed(futures):
            mix_id, verfahren, ns, hs, dauer, eps_rek, anteil_ns = future.result()
            eintrag = eintraege[mix_id]

            ziel_ns = getrennt_dir / verfahren / f"{mix_id}_Ns.wav"
            ziel_hs = getrennt_dir / verfahren / f"{mix_id}_Hs.wav"
            ziel_ns.parent.mkdir(parents=True, exist_ok=True)
            sf.write(ziel_ns, ns, SR_ZIEL, subtype="FLOAT")
            sf.write(ziel_hs, hs, SR_ZIEL, subtype="FLOAT")

            schreiber.writerow({
                "mix_id": mix_id,
                "nutzschall": eintrag["nutzschall"],
                "stoerquelle": eintrag["stoerquelle"],
                "verfahren": verfahren,
                "dauer_s": round(dauer, 3),
                "eps_rek": round(eps_rek, 6),
                "anteil_ns": round(anteil_ns, 4),
                "lib_hash": aktueller_hash,
            })
            f.flush()

            erledigt += 1
            if erledigt % 20 == 0 or erledigt == len(aufgaben):
                print(f"[{erledigt:4d}/{len(aufgaben)}] {mix_id} {verfahren:22s} "
                      f"({time.time() - t_start:.0f}s)")

    print(f"\n{erledigt} Trennungen berechnet in {time.time() - t_start:.0f}s.")
    print(f"Protokoll: {protokoll}")


if __name__ == "__main__":
    # Windows startet Worker-Prozesse per "spawn" und fuehrt dabei dieses
    # Modul erneut aus -- ohne den __main__-Schutz wuerde jeder Worker
    # versuchen, seinerseits einen Pool zu starten.
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
