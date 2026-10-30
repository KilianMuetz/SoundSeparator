"""
mimii_trennstufe.py — Hilft die Trennstufe? MIMII, roh gegen getrennt.

Hauptfrage (Ticket #468): Wirkt die Trennstufe so, wie die Box heute rechnet?
Verglichen werden Spektralsubtraktion (Empfehlung der Masterarbeit) und HPSS.
  box  = 20 MFCC, Hoechstwert je Segment (wie auf der Box)
  mel  = 128 Log-Mel-Baender, Mittelwert je Segment (bester Merkmalssatz am Luefter)

Mel-Spektrum und MFCC exakt wie auf der Box (Parameter aus feature_schema.py im Box-Code),
Detektor wie auf der Box: Skalierung auf -1 bis 1, Isolation Forest.

Aufteilung wie bei Purohit et al.: Test sind alle Fehleraufnahmen und
gleich viele Normalaufnahmen, Training sind die uebrigen Normalaufnahmen.
Je Aufnahme Kanal 1 und die ersten 5 s, wie ein Segment von SIPREMA.
Ergebnis: ergebnisse/trennstufe_<maschine>.csv
"""

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from trennung import hpss, spektralsubtraktion

MASCHINE = "pump"   # "fan" oder "pump"

BASE = Path(__file__).parent.parent
DATEN = BASE / "data" / MASCHINE
ERGEBNISSE = BASE / "ergebnisse"
BOX = BASE.parent / "siprema" / "box" / "backend"
SR = 16000
DAUER = 5 * SR

# Parameter des Mel-Spektrums direkt aus dem Box-Code
sys.path.insert(0, str(BOX))
from src import feature_schema as fs  # noqa: E402


def laden(datei):
    """Kanal 1, erste 5 s."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True, frames=DAUER)
    assert sr == SR, f"{datei}: Abtastrate {sr}"
    return y[:, 0]


def merkmale(y):
    """MFCC-Hoechstwert (wie Box) und Log-Mel-Mittel. Die Box rechnet mit 16-bit-Werten."""
    x = np.clip(y * 32767, -32768, 32767).astype(np.int16).astype(np.float64)
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=fs.FRAME_SIZE, hop_length=fs.HOP_SIZE,
                                         n_mels=fs.N_MELS, fmin=fs.MEL_FMIN, norm=fs.MEL_NORM)
    mel_db = librosa.power_to_db(mel, top_db=fs.MFCC_TOP_DB)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=fs.N_MFCC,
                                dct_type=fs.MFCC_DCT_TYPE, norm=fs.MFCC_NORM)
    return {"box": mfcc.max(axis=1), "mel": mel_db.mean(axis=1)}


def auc(X_train, X_test, label):
    """Skalierung und Isolation Forest wie auf der Box, AUC gemittelt ueber 5 Startwerte."""
    skala = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    X_train, X_test = skala.transform(X_train), skala.transform(X_test)
    werte = []
    for seed in range(5):
        wald = IsolationForest(contamination=0.01, random_state=seed).fit(X_train)
        werte.append(roc_auc_score(label, -wald.score_samples(X_test)))
    return np.mean(werte)


zeilen = [["maschine", "id", "verfahren", "box", "mel"]]

for maschine in sorted(DATEN.iterdir()):
    normal = sorted((maschine / "normal").glob("*.wav"))
    fehler = sorted((maschine / "abnormal").glob("*.wav"))

    # Test: alle Fehler und gleich viele Normalaufnahmen, Rest ist Training
    train = normal[len(fehler):]
    test = normal[:len(fehler)] + fehler
    label = [0] * len(fehler) + [1] * len(fehler)

    for verfahren, verarbeiten in [("roh", lambda y: y),
                                   ("speksub", lambda y: spektralsubtraktion(y, SR)[0]),
                                   ("hpss", lambda y: hpss(y, SR)[0])]:
        M_train = [merkmale(verarbeiten(laden(d))) for d in train]
        M_test = [merkmale(verarbeiten(laden(d))) for d in test]

        werte = [round(auc(np.array([m[k] for m in M_train]),
                           np.array([m[k] for m in M_test]), label), 3)
                 for k in ["box", "mel"]]

        zeilen.append([MASCHINE, maschine.name, verfahren, *werte])
        print(f"{maschine.name}  {verfahren:9s}  box {werte[0]:.3f}   mel {werte[1]:.3f}")

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / f"trennstufe_{MASCHINE}.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
