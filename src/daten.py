"""daten.py — Pfade, Manifest und Laden der Signale."""

import json
from functools import lru_cache
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

import mix_signals as ms

BASE = Path(__file__).parent.parent
MIX = BASE / "data" / "mixed"
GETRENNT = BASE / "ergebnisse" / "getrennt"
PROTOKOLL = BASE / "ergebnisse" / "protokoll"
SR = 16000   # Abtastrate von SIPREMA

manifest = json.loads((MIX / "manifest.json").read_text(encoding="utf-8"))
train = [i for i, e in enumerate(manifest) if e["rolle"] == "train"]
test = [i for i, e in enumerate(manifest) if e["rolle"] == "test"]

laden = lru_cache(ms.load_mono)


def auf_16khz(y):
    return resample_poly(y, 160, 441)


def mischsignal(e):
    return auf_16khz(sf.read(MIX / e["file"], dtype="float64")[0])


def ground_truth(e):
    """Reiner Nutz- und Stoerschall des Mischsignals."""
    stoer_file, art = ms.STOERQUELLEN[e["stoerquelle"]]
    nutz, stoer = ms.bestandteile(laden(ms.NUTZSCHALL[e["nutzschall"]]),
                                  laden(stoer_file), art, e["offset_s"])
    return auf_16khz(nutz), auf_16khz(stoer)


def nutzsignal(verfahren, e):
    """Nutzschall eines Verfahrens. 'roh' ist das Mischsignal, 'ground_truth' der reine Nutzschall."""
    if verfahren == "roh":
        return mischsignal(e)
    if verfahren == "ground_truth":
        return ground_truth(e)[0]
    return sf.read(GETRENNT / verfahren / f"{e['mix_id']}_Ns.wav", dtype="float64")[0]
