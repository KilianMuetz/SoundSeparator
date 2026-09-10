"""
sisdr.py — Trennqualitaet gegen die Ground Truth.

Fuer jedes Mischsignal werden das reine Nutz- und Stoersignal deterministisch
aus den Rohaufnahmen rekonstruiert. Daraus drei Groessen je Verfahren:

  si_sdr      — Signal-to-Distortion Ratio des geschaetzten Nutzsignals in dB
                (skaleninvariant, Le Roux u.a. 2019)
  si_sdri     — Verbesserung gegenueber dem unbearbeiteten Mischsignal in dB.
                Positiv heisst Gewinn, negativ heisst Schaden.
  signalerhalt / stoerrest
              — Zerlegung des geschaetzten Nutzsignals in Nutz- und Stoeranteil
                per kleinster Quadrate. Im Mischsignal sind beide gleich 1.
                Ideal waere Signalerhalt 1 bei Stoerrest 0.

Aufruf:  python src/sisdr.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import verfahren_lib as vl

sys.path.insert(0, str(Path(__file__).parent / "pre"))
import mix_signals as ms

# --- Pfade ---
BASE = Path(__file__).parent.parent
set_dir = BASE / "data" / "set"
mix_dir = BASE / "data" / "mixed"
getrennt_dir = BASE / "ergebnisse" / "getrennt"
protokoll_dir = BASE / "ergebnisse" / "protokoll"

SR = 16000
SR_QUELLE = 44100
EPS = 1e-12

_cache = {}


def roh_laden(name):
    if name not in _cache:
        _cache[name] = ms.load_mono(set_dir / name, SR_QUELLE)
    return _cache[name]


def bestandteile(eintrag):
    """Reines Nutz- und Stoersignal des Mischsignals, bei 16 kHz."""
    nutz_file = ms.NUTZSCHALL[eintrag["nutzschall"]][0]
    stoer_file, stoer_type = ms.STOERQUELLEN[eintrag["stoerquelle"]]
    offset = eintrag["offset_s"]
    sr = SR_QUELLE

    nutz = ms.trim_window(roh_laden(nutz_file), sr, ms.TARGET_DURATION, offset=offset)

    stoer_full = roh_laden(stoer_file)
    stoer = np.zeros(int(ms.TARGET_DURATION * sr))
    onset_n = int(ms.ONSET_S * sr)
    rest_n = len(stoer) - onset_n
    if stoer_type == "continuous":
        nutzbar = len(stoer_full) / sr - rest_n / sr
        stoer_offset = offset % nutzbar if nutzbar > 0 else 0.0
        stoer[onset_n:] = ms.trim_window(stoer_full, sr, rest_n / sr, offset=stoer_offset)
    else:
        event_n = min(len(stoer_full), rest_n)
        stoer[onset_n:onset_n + event_n] = stoer_full[:event_n]

    stoer = stoer * (ms.rms(nutz) / (10 ** (ms.SNR_DB / 20)) / ms.rms(stoer))
    peak = np.max(np.abs(nutz + stoer))
    faktor = ms.TARGET_PEAK / peak if peak > ms.TARGET_PEAK else 1.0

    return (resample_poly(nutz * faktor, SR // 100, sr // 100),
            resample_poly(stoer * faktor, SR // 100, sr // 100))


def si_sdr(schaetzung, ziel):
    """Skaleninvariantes SDR in dB."""
    ziel = ziel - ziel.mean()
    schaetzung = schaetzung - schaetzung.mean()
    anteil = np.dot(schaetzung, ziel) / (np.dot(ziel, ziel) + EPS) * ziel
    rest = schaetzung - anteil
    return 10 * np.log10((np.dot(anteil, anteil) + EPS) / (np.dot(rest, rest) + EPS))


def zerlegung(schaetzung, nutz, stoer):
    """Kleinste Quadrate: schaetzung ~ a*nutz + b*stoer."""
    A = np.column_stack([nutz, stoer])
    a, b = np.linalg.lstsq(A, schaetzung, rcond=None)[0]
    return float(a), float(b)


def signal_laden(verfahren, eintrag):
    if verfahren == "roh":
        y, sr = sf.read(mix_dir / eintrag["file"], dtype="float64")
        return resample_poly(y, SR // 100, sr // 100)
    y, _ = sf.read(getrennt_dir / verfahren / f"{eintrag['mix_id']}_Ns.wav",
                   dtype="float64")
    return y


with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"{len(manifest)} Mischsignale, Ground Truth wird rekonstruiert ...\n")
gt = [bestandteile(e) for e in manifest]
basis = [si_sdr(n + s, n) for n, s in gt]     # SI-SDR des Mischsignals

zeilen = []
for verfahren in ["roh"] + list(vl.VERFAHREN):
    werte, gewinne = [], []
    for e, (nutz, stoer), b in zip(manifest, gt, basis):
        ns = signal_laden(verfahren, e)
        m = min(len(ns), len(nutz))
        ns, n_, s_ = ns[:m], nutz[:m], stoer[:m]

        wert = si_sdr(ns, n_)
        erhalt, rest = zerlegung(ns, n_, s_)
        werte.append(wert)
        gewinne.append(wert - b)

        zeilen.append({
            "mix_id": e["mix_id"],
            "nutzschall": e["nutzschall"],
            "stoerquelle": e["stoerquelle"],
            "verfahren": verfahren,
            "si_sdr": round(wert, 3),
            "si_sdri": round(wert - b, 3),
            "signalerhalt": round(erhalt, 4),
            "stoerrest": round(rest, 4),
        })

    rs = [z for z in zeilen if z["verfahren"] == verfahren]
    print(f"{verfahren:22s} SI-SDR {np.median(werte):7.2f} dB   "
          f"Gewinn {np.median(gewinne):+6.2f} dB   "
          f"Erhalt {np.median([z['signalerhalt'] for z in rs]):5.2f}   "
          f"Stoerrest {np.median([z['stoerrest'] for z in rs]):5.2f}")

protokoll_dir.mkdir(parents=True, exist_ok=True)
ziel = protokoll_dir / "trennqualitaet.csv"
with open(ziel, "w", newline="", encoding="utf-8") as f:
    schreiber = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()))
    schreiber.writeheader()
    schreiber.writerows(zeilen)

print(f"\nProtokoll: {ziel}")
