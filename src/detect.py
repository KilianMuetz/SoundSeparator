"""detect.py — Anomalieerkennung mit einem Isolation Forest je Verfahren.

'roh' ist die untere Marke, 'ground_truth' die obere.
"""

import csv

import librosa
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import verfahren_lib as vl
from daten import PROTOKOLL, SR, manifest, nutzsignal, test, train


def merkmale(y):
    """20 MFCC je Fenster von 25 ms, Versatz 10 ms, bei einheitlicher Lautstaerke."""
    y = y / np.sqrt(np.mean(y ** 2))
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512,
                                win_length=400, hop_length=160).T


label = np.array([e["nutzschall"] != "normal" for e in manifest])
wahr = label[test]

scores = [["mix_id", "verfahren", "score"]]
aucs = [["verfahren", "auc", "auc_std"]]
metriken = [["verfahren", "schwelle", "tp", "fp", "fn", "tn",
             "genauigkeit", "praezision", "tpr"]]

for verfahren in ["roh", *vl.VERFAHREN, "ground_truth"]:
    X = [merkmale(nutzsignal(verfahren, e)) for e in manifest]
    X_train = np.vstack([X[i] for i in train])

    S = []   # Anomaliescore je Startwert und Segment
    for seed in range(5):
        wald = IsolationForest(n_estimators=100, random_state=seed).fit(X_train)
        S.append([np.mean(-wald.score_samples(x)) for x in X])
    S = np.array(S)
    auc = [roc_auc_score(wahr, s[test]) for s in S]
    score = S.mean(axis=0)

    schwelle = np.percentile(score[train], 90)   # wie Dohi et al. 2022
    alarm = score[test] > schwelle
    tp, fp = np.sum(alarm & wahr), np.sum(alarm & ~wahr)
    fn, tn = np.sum(~alarm & wahr), np.sum(~alarm & ~wahr)

    aucs.append([verfahren, round(np.mean(auc), 4), round(np.std(auc), 4)])
    metriken.append([verfahren, round(schwelle, 4), tp, fp, fn, tn,
                     round((tp + tn) / len(wahr), 3), round(tp / max(tp + fp, 1), 3),
                     round(tp / (tp + fn), 3)])
    scores += [[manifest[i]["mix_id"], verfahren, round(score[i], 6)] for i in test]
    print(f"{verfahren:22s} AUC = {np.mean(auc):.4f} +- {np.std(auc):.4f}")

for name, zeilen in [("erkennung.csv", scores), ("erkennung_auc.csv", aucs),
                     ("erkennung_metriken.csv", metriken)]:
    with open(PROTOKOLL / name, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(zeilen)
