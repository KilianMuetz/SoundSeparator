"""detect.py — je Trennverfahren ein Isolation Forest auf Normaldaten."""

import csv
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import verfahren_lib as vl
from io_utils import aufteilen, signal_laden

BASE = Path(__file__).parent.parent
mix_dir = BASE / "data" / "mixed"
getrennt_dir = BASE / "ergebnisse" / "getrennt"
protokoll_dir = BASE / "ergebnisse" / "protokoll"


def merkmale(y):
    """20 MFCC je Fenster von 25 ms, Versatz 10 ms, bei einheitlicher Lautstaerke."""
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=512,
                             win_length=400, hop_length=160)
    return m.T.astype(np.float64)


with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)
train, test = aufteilen(manifest)
label = [e["nutzschall"] != "normal" for e in manifest]

scores, aucs = [["mix_id", "verfahren", "score"]], [["verfahren", "auc", "auc_std"]]
for verfahren in ["roh"] + list(vl.VERFAHREN):
    X = [merkmale(signal_laden(verfahren, e, mix_dir, getrennt_dir, 16000, 44100))
         for e in manifest]
    X_train = np.vstack([X[i] for i in train])

    je_seed = []
    for seed in range(5):
        modell = IsolationForest(n_estimators=100, random_state=seed).fit(X_train)
        je_seed.append([np.mean(-modell.score_samples(X[i])) for i in test])
    auc = [roc_auc_score([label[i] for i in test], s) for s in je_seed]

    aucs.append([verfahren, round(np.mean(auc), 4), round(np.std(auc), 4)])
    for i, s in zip(test, np.mean(je_seed, axis=0)):
        scores.append([manifest[i]["mix_id"], verfahren, round(s, 6)])
    print(f"{verfahren:22s} AUC = {np.mean(auc):.4f} +- {np.std(auc):.4f}")

for name, zeilen in [("erkennung.csv", scores), ("erkennung_auc.csv", aucs)]:
    with open(protokoll_dir / name, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(zeilen)
