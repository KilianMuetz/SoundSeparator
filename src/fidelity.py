"""
fidelity.py — Score-Treue: wie nah kommt das getrennte Signal an die Ground Truth?

Fuer jedes Mischsignal wird das reine Nutzsignal deterministisch aus den
Rohaufnahmen rekonstruiert (Ground Truth). Bewertet wird der Abstand zwischen
diesem reinen Signal und dem geschaetzten Nutzsignal jedes Verfahrens.

Referenz und Messwert stammen aus demselben Mischsignal, derselben Aufnahme
und derselben Sekunde. Aufnahmesitzung, Drehzahl und Mikrofonposition sind
damit identisch und kuerzen sich heraus. Das unterscheidet diese Groesse von
der AUC, die normale und anomale Segmente ueber verschiedene Aufnahmen hinweg
vergleicht.

Zwei Groessen je Mischsignal und Verfahren:
  merkmalsabstand — mittlerer euklidischer Abstand der MFCC-Frames zur
                    Ground Truth, im standardisierten Merkmalsraum
  score_fehler    — Betrag der Abweichung des Anomaliescores gegenueber der
                    Ground Truth, bewertet mit dem Oracle-Modell

Beide werden zusaetzlich als Verhaeltnis zur unbearbeiteten Referenz "roh"
ausgewiesen. Ein Wert unter 1 bedeutet eine Verbesserung durch die Trennung.

Aufruf:  python src/fidelity.py
"""

import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import verfahren_lib as vl

sys.path.insert(0, str(Path(__file__).parent / "pre"))
import mix_signals as ms

# --- Pfade ---
BASE = Path(__file__).parent.parent
set_dir = BASE / "data" / "set"
mix_dir = BASE / "data" / "mixed"
getrennt_dir = BASE / "ergebnisse" / "getrennt"
protokoll_dir = BASE / "ergebnisse" / "protokoll"

# --- Parameter (identisch zu detect.py) ---
SR = 16000
SR_QUELLE = 44100
N_MFCC = 20
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
SEEDS = [0, 1, 2, 3, 4]

_cache = {}


def roh_laden(name):
    """Rohaufnahme mit Zwischenspeicher, die Normalaufnahme ist gross."""
    if name not in _cache:
        _cache[name] = ms.load_mono(set_dir / name, SR_QUELLE)
    return _cache[name]


def ground_truth(eintrag):
    """Reines Nutzsignal des Mischsignals, inklusive der Mischskalierung."""
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

    # Skalierung exakt wie in mix_at_snr
    ziel_rms = ms.rms(nutz) / (10 ** (ms.SNR_DB / 20))
    mix = nutz + stoer * (ziel_rms / ms.rms(stoer))
    peak = np.max(np.abs(mix))
    faktor = ms.TARGET_PEAK / peak if peak > ms.TARGET_PEAK else 1.0

    return resample_poly(nutz * faktor, SR // 100, sr // 100)


def merkmale(y):
    """Pegelnormiert, da der Energieanteil bereits separat ausgewiesen wird."""
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
                             win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    return m.T.astype(np.float64)


def signal_laden(verfahren, eintrag):
    if verfahren == "roh":
        y, sr = sf.read(mix_dir / eintrag["file"], dtype="float64")
        return resample_poly(y, SR // 100, sr // 100)
    y, _ = sf.read(getrennt_dir / verfahren / f"{eintrag['mix_id']}_Ns.wav",
                   dtype="float64")
    return y


# --- Mischsignale ---
with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

train_idx = [i for i, e in enumerate(manifest) if e["rolle"] == "train"]
print(f"{len(manifest)} Mischsignale, {len(train_idx)} davon Trainingssegmente.\n")

# --- Ground Truth und Oracle-Modell ---
print("Ground Truth wird rekonstruiert ...")
X_gt = [merkmale(ground_truth(e)) for e in manifest]

X_train = np.vstack([X_gt[i] for i in train_idx])
skalierer = StandardScaler().fit(X_train)
X_train_s = skalierer.transform(X_train)

oracle = []
for seed in SEEDS:
    m = IsolationForest(random_state=seed, n_estimators=100)
    m.fit(X_train_s)
    oracle.append(m)


def bewerte(X):
    """Anomaliescore im Merkmalsraum des Oracle-Modells, ueber Seeds gemittelt."""
    Xs = skalierer.transform(X)
    return float(np.mean([np.mean(-m.score_samples(Xs)) for m in oracle]))


score_gt = [bewerte(x) for x in X_gt]
print("Oracle-Modell trainiert.\n")

# --- Bewertung je Verfahren ---
zeilen = []
for verfahren in ["roh"] + list(vl.VERFAHREN):
    for e, xg, sg in zip(manifest, X_gt, score_gt):
        x = merkmale(signal_laden(verfahren, e))
        n = min(len(x), len(xg))

        abstand = float(np.mean(np.linalg.norm(
            skalierer.transform(x[:n]) - skalierer.transform(xg[:n]), axis=1)))
        fehler = abs(bewerte(x) - sg)

        zeilen.append({
            "mix_id": e["mix_id"],
            "nutzschall": e["nutzschall"],
            "stoerquelle": e["stoerquelle"],
            "verfahren": verfahren,
            "merkmalsabstand": round(abstand, 4),
            "score_fehler": round(fehler, 6),
            "score_gt": round(sg, 6),
        })
    werte = [z for z in zeilen if z["verfahren"] == verfahren]
    print(f"{verfahren:24s} Merkmalsabstand {np.median([z['merkmalsabstand'] for z in werte]):7.3f}   "
          f"Score-Fehler {np.median([z['score_fehler'] for z in werte]):.5f}")

# --- Verhaeltnis zur unbearbeiteten Referenz ---
roh_ab = {z["mix_id"]: z["merkmalsabstand"] for z in zeilen if z["verfahren"] == "roh"}
roh_sf = {z["mix_id"]: z["score_fehler"] for z in zeilen if z["verfahren"] == "roh"}
for z in zeilen:
    z["abstand_rel"] = round(z["merkmalsabstand"] / (roh_ab[z["mix_id"]] + 1e-12), 4)
    z["score_fehler_rel"] = round(z["score_fehler"] / (roh_sf[z["mix_id"]] + 1e-12), 4)

protokoll_dir.mkdir(parents=True, exist_ok=True)
ziel = protokoll_dir / "treue.csv"
with open(ziel, "w", newline="", encoding="utf-8") as f:
    schreiber = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()))
    schreiber.writeheader()
    schreiber.writerows(zeilen)

print(f"\nProtokoll: {ziel}")
