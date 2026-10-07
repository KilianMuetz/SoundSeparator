"""
detect.py — Anomalieerkennung je Trennverfahren.

Je Verfahren ein Isolation Forest, trainiert nur auf den Normalsegmenten des
Trainingsbereichs, die dasselbe Verfahren durchlaufen haben. "roh" ist das
unbearbeitete Mischsignal als Referenz.

Aufruf:  python src/detect.py
"""

import csv
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import verfahren_lib as vl
from io_utils import signal_laden

BASE = Path(__file__).parent.parent
mix_dir = BASE / "data" / "mixed"
getrennt_dir = BASE / "ergebnisse" / "getrennt"
protokoll_dir = BASE / "ergebnisse" / "protokoll"

SR = 16000
SR_QUELLE = 44100
SEEDS = [0, 1, 2, 3, 4]


def merkmale(y):
    """20 MFCC je Fenster (25 ms, Versatz 10 ms), Ergebnis (Fenster, 20)."""
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12)    # einheitliche Lautstaerke
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512,
                             win_length=400, hop_length=160)
    return m.T.astype(np.float64)


with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

train = [i for i, e in enumerate(manifest) if e["rolle"] == "train"]
test = [i for i, e in enumerate(manifest) if e["rolle"] == "test"]
label = [0 if e["nutzschall"] == "normal" else 1 for e in manifest]

zeilen, auc_zeilen = [], []
for verfahren in ["roh"] + list(vl.VERFAHREN):
    X = [merkmale(signal_laden(verfahren, e, mix_dir, getrennt_dir, SR, SR_QUELLE))
         for e in manifest]
    X_train = np.vstack([X[i] for i in train])

    scores, aucs = [], []
    for seed in SEEDS:
        modell = IsolationForest(n_estimators=100, random_state=seed).fit(X_train)
        s = [np.mean(-modell.score_samples(X[i])) for i in test]  # hoch = auffaellig
        scores.append(s)
        aucs.append(roc_auc_score([label[i] for i in test], s))
    score = np.mean(scores, axis=0)

    auc_zeilen.append({"verfahren": verfahren,
                       "auc": round(float(np.mean(aucs)), 4),
                       "auc_std": round(float(np.std(aucs)), 4)})
    for i, sc in zip(test, score):
        e = manifest[i]
        zeilen.append({"mix_id": e["mix_id"], "nutzschall": e["nutzschall"],
                       "stoerquelle": e["stoerquelle"], "offset_s": e["offset_s"],
                       "verfahren": verfahren, "label": label[i],
                       "score": round(float(sc), 6)})
    print(f"{verfahren:22s} AUC = {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")

protokoll_dir.mkdir(parents=True, exist_ok=True)
for name, daten in [("erkennung.csv", zeilen), ("erkennung_auc.csv", auc_zeilen)]:
    with open(protokoll_dir / name, "w", newline="", encoding="utf-8") as f:
        schreiber = csv.DictWriter(f, fieldnames=list(daten[0]))
        schreiber.writeheader()
        schreiber.writerows(daten)
