"""
mimii_mel.py — Log-Mel-Baender als Merkmal, roh und nach der Trennung.

MFCC fassen jedes Spektrum in 20 grobe Zahlen. Log-Mel-Baender behalten die
feine spektrale Form: 128 Baender, je Band Mittelwert ueber das Segment.
Die Streuung je Band zeigt zusaetzlich, wie stark ein Band schwankt,
etwa durch periodisches Klopfen.

  mittel        = 128 Werte, Mittelwert je Band
  mittel_streu  = 256 Werte, Mittelwert und Standardabweichung je Band

Mel-Spektrum exakt wie auf der Box (Parameter aus feature_schema.py im Box-Code),
Detektor wie auf der Box: Skalierung auf -1 bis 1, Isolation Forest.
Daten und Aufteilung wie in mimii.py.
Ergebnis: ergebnisse/mimii_mel.csv
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

from trennung import spektralsubtraktion

BASE = Path(__file__).parent.parent
DATEN = BASE / "data" / "fan"
ERGEBNISSE = BASE / "ergebnisse"
BOX = BASE.parent / "siprema" / "box" / "backend"
SR = 16000
DAUER = 5 * SR

# Parameter des Mel-Spektrums direkt aus dem Box-Code
sys.path.insert(0, str(BOX))
from src import feature_schema as fs  # noqa: E402

ZUSAMMENFASSUNG = {
    "mittel": lambda m: m.mean(axis=1),
    "mittel_streu": lambda m: np.concatenate([m.mean(axis=1), m.std(axis=1)]),
}


def laden(datei):
    """Kanal 1, erste 5 s."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True, frames=DAUER)
    assert sr == SR, f"{datei}: Abtastrate {sr}"
    return y[:, 0]


def log_mel(y):
    """Log-Mel-Spektrum je Zeitfenster wie auf der Box. Die Box rechnet mit 16-bit-Werten."""
    x = np.clip(y * 32767, -32768, 32767).astype(np.int16).astype(np.float64)
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=fs.FRAME_SIZE, hop_length=fs.HOP_SIZE,
                                         n_mels=fs.N_MELS, fmin=fs.MEL_FMIN, norm=fs.MEL_NORM)
    return librosa.power_to_db(mel, top_db=fs.MFCC_TOP_DB)


def auc(X_train, X_test, label):
    """Skalierung und Isolation Forest wie auf der Box, AUC gemittelt ueber 5 Startwerte."""
    skala = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    X_train, X_test = skala.transform(X_train), skala.transform(X_test)
    werte = []
    for seed in range(5):
        wald = IsolationForest(contamination=0.01, random_state=seed).fit(X_train)
        werte.append(roc_auc_score(label, -wald.score_samples(X_test)))
    return np.mean(werte)


zeilen = [["maschine", "verfahren", *ZUSAMMENFASSUNG]]

for maschine in sorted(DATEN.iterdir()):
    normal = sorted((maschine / "normal").glob("*.wav"))
    fehler = sorted((maschine / "abnormal").glob("*.wav"))

    # Test: alle Fehler und gleich viele Normalaufnahmen, Rest ist Training
    train = normal[len(fehler):]
    test = normal[:len(fehler)] + fehler
    label = [0] * len(fehler) + [1] * len(fehler)

    for verfahren, verarbeiten in [("roh", lambda y: y),
                                   ("getrennt", lambda y: spektralsubtraktion(y, SR)[0])]:
        S_train = [log_mel(verarbeiten(laden(d))) for d in train]
        S_test = [log_mel(verarbeiten(laden(d))) for d in test]

        werte = []
        for f in ZUSAMMENFASSUNG.values():
            X_train = np.array([f(s) for s in S_train])
            X_test = np.array([f(s) for s in S_test])
            werte.append(round(auc(X_train, X_test, label), 3))

        zeilen.append([maschine.name, verfahren, *werte])
        print(f"{maschine.name}  {verfahren:9s}  " +
              "   ".join(f"{n} {w:.3f}" for n, w in zip(ZUSAMMENFASSUNG, werte)))

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / "mimii_mel.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
