"""
feld.py — Wie gut trennt der Detektor Stoersegmente vom Normalzustand, roh und nach der Trennung.

Je Fall in NOISE/ liegen Normalsegmente und Stoersegmente derselben Box.
Ein Isolation Forest lernt den Normalzustand und vergibt jedem Segment einen Score.

Mass ist die AUC zwischen Normal- und Stoersegmenten:
  1,0 = jedes Stoersegment wirkt anomaler als jedes Normalsegment
  0,5 = Stoersegmente sind vom Normalzustand nicht zu unterscheiden
Sinkt die AUC nach der Trennung, wurde der Stoerschall entfernt.

Die Normalsegmente sind zugleich Trainingsdaten. Ihre Scores sind dadurch zu
niedrig, die AUC zu hoch. Das trifft roh und getrennt gleich, der Vergleich bleibt fair.

Merkmale wie in der Masterarbeit (detect.py).
Ergebnis: ergebnisse/feld.csv
"""

import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from trennung import spektralsubtraktion

BASE = Path(__file__).parent.parent
NOISE = BASE / "NOISE"
ERGEBNISSE = BASE / "ergebnisse"
SR = 16000


def laden(datei):
    """Mono, 16 kHz. Andere Abtastrate: umrechnen."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True)
    y = y[:, 0]
    if sr != SR:
        y = resample_poly(y, SR, sr)
    return y


def segmente(ordner):
    """Alle Box-Segmente eines Ordners. MIMII-Clips (Name nur Ziffern) bleiben draussen."""
    return [laden(d) for d in sorted(ordner.glob("*.wav")) if not d.stem.isdigit()]


def merkmale(y):
    """20 MFCC je Fenster von 25 ms, Versatz 10 ms, bei einheitlicher Lautstaerke."""
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=512,
                                win_length=400, hop_length=160).T


def auc(normal, stoer):
    """AUC Normal gegen Stoerung, gemittelt ueber 5 Startwerte."""
    X_train = np.vstack([merkmale(y) for y in normal])
    label = [0] * len(normal) + [1] * len(stoer)
    werte = []
    for seed in range(5):
        wald = IsolationForest(n_estimators=100, random_state=seed).fit(X_train)
        scores = [np.mean(-wald.score_samples(merkmale(y))) for y in normal + stoer]
        werte.append(roc_auc_score(label, scores))
    return np.mean(werte)


zeilen = [["fall", "n_normal", "n_stoer", "auc_roh", "auc_getrennt"]]

for fall in sorted(NOISE.iterdir()):
    normal = segmente(fall / "normal")
    stoer = segmente(fall / "noise")

    roh = auc(normal, stoer)
    getrennt = auc([spektralsubtraktion(y, SR)[0] for y in normal],
                   [spektralsubtraktion(y, SR)[0] for y in stoer])

    zeilen.append([fall.name, len(normal), len(stoer), round(roh, 3), round(getrennt, 3)])
    print(f"{fall.name}  normal {len(normal):3d}  stoer {len(stoer):3d}   "
          f"AUC roh {roh:.3f}   getrennt {getrennt:.3f}")

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / "feld.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
