"""
mix_signals.py — Erzeugt die Mischsignale aus Nutzschall- und Stoerschallaufnahmen.

Jedes Segment ist 5 s lang. Der Stoerschall setzt nach 2 s ein, wird ueber das
ganze Segment auf 0 dB SNR skaliert, danach wird der Spitzenwert auf 0,9 begrenzt.

Aufruf:  python src/mix_signals.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).parent.parent
set_dir = BASE / "data" / "set"
mix_dir = BASE / "data" / "mixed"

SR = 44100
DAUER = 5.0      # Segmentlaenge in s
EINSATZ = 2.0    # Einsatz des Stoerschalls in s
SNR_DB = 0.0     # Nutz- und Stoerschall gleich laut, ueber das ganze Segment
PEAK = 0.9       # Obergrenze gegen Uebersteuerung

NORMAL_TRAIN = [0.02, 0.14, 0.26, 0.38]   # Training in den ersten 40 % der Normalaufnahme
NORMAL_TEST = [0.66, 0.80, 0.94]          # Test in den letzten 35 %
ANOMALIE_OFFSETS = [0.0, 5.0]             # zwei Segmente je Anomalieaufnahme in s

NUTZSCHALL = {
    "normal":      "normalzustand.wav",
    "a1_leicht":   "gewicht_leicht.wav",
    "a1_stark":    "gewicht_schwer.wav",
    "a2_leicht":   "locker_leicht.wav",
    "a2_deutlich": "locker_schwer.wav",
    "a3_leicht":   "tape_leicht.wav",
    "a3_deutlich": "tape_schwer.wav",
}

# "lang": laengere Aufnahme, Ausschnitt an wechselnder Stelle
# "esc":  5-s-Ausschnitt aus ESC-50, eingesetzt ab seinem Beginn
STOERQUELLEN = {
    "tuerschlaege":   ("tuerschlaege.wav", "lang"),
    "metallhammer":   ("metallhammer.wav", "lang"),
    "gespraech":      ("gespraech_6dzb.wav", "lang"),
    "kraehe":         ("kraehe.wav", "esc"),
    "voegel":         ("voegel.wav", "esc"),
    "kuh":            ("kuh.wav", "esc"),
    "alarm":          ("alarm.wav", "esc"),
    "hupe":           ("hupe.wav", "esc"),
    "radio":          ("radio_6dzb.wav", "lang"),
    "verkehr":        ("verkehr.wav", "lang"),
    "akku_konstant":  ("makita_konstant.wav", "lang"),
    "akku_dynamisch": ("makita_dynamisch.wav", "lang"),
}


def load_mono(name):
    y, sr = sf.read(set_dir / name, always_2d=True)
    assert sr == SR, f"{name}: Abtastrate {sr} != {SR}"
    return y.mean(axis=1)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def bestandteile(nutz_full, stoer_full, art, offset):
    """Nutz- und Stoeranteil eines Mischsignals. Das Mischsignal ist ihre Summe."""
    n, einsatz = int(DAUER * SR), int(EINSATZ * SR)
    rest = n - einsatz

    start = int(offset * SR)
    nutz = nutz_full[start:start + n]
    assert len(nutz) == n, "Nutzschallaufnahme zu kurz"

    stoer = np.zeros(n)
    if art == "lang":
        nutzbar = len(stoer_full) / SR - rest / SR
        s = int((offset % nutzbar if nutzbar > 0 else 0.0) * SR)
        stoer[einsatz:] = stoer_full[s:s + rest]
    else:
        k = min(len(stoer_full), rest)
        stoer[einsatz:einsatz + k] = stoer_full[:k]

    stoer = stoer * (rms(nutz) / (10 ** (SNR_DB / 20)) / rms(stoer))
    faktor = min(1.0, PEAK / np.max(np.abs(nutz + stoer)))
    return nutz * faktor, stoer * faktor


if __name__ == "__main__":
    mix_dir.mkdir(parents=True, exist_ok=True)
    spanne = sf.info(set_dir / NUTZSCHALL["normal"]).duration - DAUER
    manifest = []

    for nutz_name, nutz_file in NUTZSCHALL.items():
        if nutz_name == "normal":
            plan = [(spanne * a, "train") for a in NORMAL_TRAIN] + \
                   [(spanne * a, "test") for a in NORMAL_TEST]
        else:
            plan = [(o, "test") for o in ANOMALIE_OFFSETS]
        nutz_full = load_mono(nutz_file)

        for stoer_name, (stoer_file, art) in STOERQUELLEN.items():
            stoer_full = load_mono(stoer_file)
            for seg, (offset, rolle) in enumerate(plan, 1):
                mix_id = f"M{len(manifest) + 1:03d}"
                datei = f"{mix_id}_{nutz_name}_{stoer_name}_s{seg}.wav"
                nutz, stoer = bestandteile(nutz_full, stoer_full, art, offset)
                sf.write(mix_dir / datei, nutz + stoer, SR, subtype="PCM_24")
                manifest.append({"mix_id": mix_id, "nutzschall": nutz_name,
                                 "stoerquelle": stoer_name, "segment": seg,
                                 "offset_s": round(offset, 2), "rolle": rolle,
                                 "snr_db": SNR_DB, "file": datei})

    with open(mix_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{len(manifest)} Mischsignale erzeugt.")
