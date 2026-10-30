"""
mimii.py — Anomalieerkennung auf MIMII, roh und nach der Trennung.

MIMII (Purohit et al. 2019): Ventilatoren bei -6 dB, Fabriklaerm im ganzen Segment.
Je Maschine (id_00 ... id_06) Aufnahmen im Normalzustand und mit Fehlern.

Aufteilung wie bei Purohit et al.: Test sind alle Fehleraufnahmen und
gleich viele Normalaufnahmen, Training sind die uebrigen Normalaufnahmen.

Je Aufnahme Kanal 1 und die ersten 5 s, wie ein Segment von SIPREMA.
Merkmale und Detektor wie in der Masterarbeit (detect.py).
Ergebnis: ergebnisse/mimii.csv
"""

import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from trennung import spektralsubtraktion

BASE = Path(__file__).parent.parent
DATEN = BASE / "data" / "fan"
ERGEBNISSE = BASE / "ergebnisse"
SR = 16000
DAUER = 5 * SR


def laden(datei):
    """Kanal 1, erste 5 s."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True, frames=DAUER)
    assert sr == SR, f"{datei}: Abtastrate {sr}"
    return y[:, 0]


def merkmale(y):
    """20 MFCC je Fenster von 25 ms, Versatz 10 ms, bei einheitlicher Lautstaerke."""
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512,
                                win_length=400, hop_length=160).T


def auc(M_train, M_test, label):
    """AUC auf dem Testteil, gemittelt ueber 5 Startwerte."""
    X_train = np.vstack(M_train)
    werte = []
    for seed in range(5):
        wald = IsolationForest(n_estimators=100, random_state=seed).fit(X_train)
        scores = [np.mean(-wald.score_samples(m)) for m in M_test]
        werte.append(roc_auc_score(label, scores))
    return np.mean(werte)


zeilen = [["maschine", "n_train", "n_test", "auc_roh", "auc_getrennt"]]

for maschine in sorted(DATEN.iterdir()):
    normal = sorted((maschine / "normal").glob("*.wav"))
    fehler = sorted((maschine / "abnormal").glob("*.wav"))

    # Test: alle Fehler und gleich viele Normalaufnahmen, Rest ist Training
    train = normal[len(fehler):]
    test = normal[:len(fehler)] + fehler
    label = [0] * len(fehler) + [1] * len(fehler)

    ergebnis = {}
    for name, verarbeiten in [("roh", lambda y: y),
                              ("getrennt", lambda y: spektralsubtraktion(y, SR)[0])]:
        M_train = [merkmale(verarbeiten(laden(d))) for d in train]
        M_test = [merkmale(verarbeiten(laden(d))) for d in test]
        ergebnis[name] = auc(M_train, M_test, label)

    zeilen.append([maschine.name, len(train), len(test),
                   round(ergebnis["roh"], 3), round(ergebnis["getrennt"], 3)])
    print(f"{maschine.name}  train {len(train):4d}  test {len(test):4d}   "
          f"AUC roh {ergebnis['roh']:.3f}   getrennt {ergebnis['getrennt']:.3f}")

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / "mimii.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
