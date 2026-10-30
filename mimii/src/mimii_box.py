"""
mimii_box.py — Anomalieerkennung auf MIMII mit den Merkmalen von SIPREMA, roh und nach der Trennung.

Merkmale: AudioFeatures aus dem Box-Code (siprema/box/backend), 209 Werte je 5-s-Segment.
Zwei Detektoren auf denselben Merkmalen:
  box   = wie auf der Box: nur die 20 MFCC, Skalierung auf -1 bis 1, Isolation Forest
  alle  = alle 209 Merkmale, sonst gleich

Daten und Aufteilung wie in mimii.py.
Die Merkmale werden in ergebnisse/merkmale/ zwischengespeichert, ein zweiter Lauf rechnet nur noch den Detektor.
Ergebnis: ergebnisse/mimii_box.csv
"""

import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from trennung import spektralsubtraktion

BASE = Path(__file__).parent.parent
DATEN = BASE / "data" / "fan"
ERGEBNISSE = BASE / "ergebnisse"
MERKMALE = ERGEBNISSE / "merkmale"
BOX = BASE.parent / "siprema" / "box" / "backend"
SR = 16000
DAUER = 5 * SR

# Merkmalscode direkt aus dem Box-Repo laden, ohne die restliche Box-Umgebung
sys.path.insert(0, str(BOX))
spec = importlib.util.spec_from_file_location(
    "audio_features", BOX / "src" / "application" / "domain" / "audio_features.py")
audio_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_features)
AudioFeatures = audio_features.AudioFeatures
MFCC = AudioFeatures.Columns.MFCC


def laden(datei):
    """Kanal 1, erste 5 s."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True, frames=DAUER)
    assert sr == SR, f"{datei}: Abtastrate {sr}"
    return y[:, 0]


def box_merkmale(y):
    """Merkmale wie auf der Box. Die Box rechnet mit 16-bit-Ganzzahlen."""
    pcm = np.clip(y * 32767, -32768, 32767).astype(np.int16).tobytes()
    return AudioFeatures.from_audio_bytes(pcm, SR, datetime.now()).to_dataframe().iloc[0]


def merkmale(dateien, name, verarbeiten):
    """Merkmale aller Dateien, aus dem Zwischenspeicher oder neu berechnet."""
    datei = MERKMALE / f"{name}.csv"
    if datei.exists():
        return pd.read_csv(datei, index_col=0)
    df = pd.DataFrame([box_merkmale(verarbeiten(laden(d))) for d in dateien],
                      index=[d.name for d in dateien])
    MERKMALE.mkdir(parents=True, exist_ok=True)
    df.to_csv(datei)
    return df


def auc(X_train, X_test, label):
    """Skalierung und Isolation Forest wie auf der Box, AUC gemittelt ueber 5 Startwerte."""
    skala = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    X_train, X_test = skala.transform(X_train), skala.transform(X_test)
    werte = []
    for seed in range(5):
        wald = IsolationForest(contamination=0.01, random_state=seed).fit(X_train)
        werte.append(roc_auc_score(label, -wald.score_samples(X_test)))
    return np.mean(werte)


zeilen = [["maschine", "verfahren", "auc_box", "auc_alle"]]

for maschine in sorted(DATEN.iterdir()):
    normal = sorted((maschine / "normal").glob("*.wav"))
    fehler = sorted((maschine / "abnormal").glob("*.wav"))

    # Test: alle Fehler und gleich viele Normalaufnahmen, Rest ist Training
    train = normal[len(fehler):]
    test = normal[:len(fehler)] + fehler
    label = [0] * len(fehler) + [1] * len(fehler)

    for verfahren, verarbeiten in [("roh", lambda y: y),
                                   ("getrennt", lambda y: spektralsubtraktion(y, SR)[0])]:
        M_train = merkmale(train, f"{maschine.name}_{verfahren}_train", verarbeiten)
        M_test = merkmale(test, f"{maschine.name}_{verfahren}_test", verarbeiten)

        # Fehlende Werte (z. B. Schiefe bei flachem Spektrum) als 0
        box = auc(M_train[MFCC].fillna(0), M_test[MFCC].fillna(0), label)
        alle = auc(M_train.fillna(0), M_test.fillna(0), label)

        zeilen.append([maschine.name, verfahren, round(box, 3), round(alle, 3)])
        print(f"{maschine.name}  {verfahren:9s}  AUC box {box:.3f}   alle {alle:.3f}")

with open(ERGEBNISSE / "mimii_box.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
