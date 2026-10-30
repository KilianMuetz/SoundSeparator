"""
mimii_physik.py — Physikalisch begruendete Merkmale fuer rotierende Maschinen, roh und nach der Trennung.

  ton    = Tonalitaet: Toene ueber dem Grundrauschen, 64 Baender bis 2 kHz,
           dazu die spektrale Flachheit (Ton oder Rauschen). 65 Werte.
  huelle = Huellkurvenspektrum: Rhythmus der Lautstaerke von 2 bis 200 Hz,
           etwa Pulsieren im Takt der Drehung. 99 Werte.
  kombi  = Log-Mel-Mittel (128) + ton + huelle. 292 Werte.

Festgelegt vor dem ersten Lauf, nicht nach Ergebnissen ausgewaehlt.
Detektor wie auf der Box: Skalierung auf -1 bis 1, Isolation Forest.
Daten und Aufteilung wie in mimii.py.
Ergebnis: ergebnisse/mimii_physik.csv
"""

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import hilbert, welch
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
EPS = 1e-12

# Parameter des Mel-Spektrums direkt aus dem Box-Code
sys.path.insert(0, str(BOX))
from src import feature_schema as fs  # noqa: E402


def laden(datei):
    """Kanal 1, erste 5 s."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True, frames=DAUER)
    assert sr == SR, f"{datei}: Abtastrate {sr}"
    return y[:, 0]


def mel_mittel(y):
    """Log-Mel-Mittel je Band wie in mimii_mel.py. 128 Werte."""
    x = np.clip(y * 32767, -32768, 32767).astype(np.int16).astype(np.float64)
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=fs.FRAME_SIZE, hop_length=fs.HOP_SIZE,
                                         n_mels=fs.N_MELS, fmin=fs.MEL_FMIN, norm=fs.MEL_NORM)
    return librosa.power_to_db(mel, top_db=fs.MFCC_TOP_DB).mean(axis=1)


def ton(y):
    """Toene ueber dem Grundrauschen. 65 Werte."""
    f, P = welch(y, fs=SR, nperseg=4096)                     # feines Spektrum, 3,9 Hz je Wert
    P_db = 10 * np.log10(P + EPS)
    grund = median_filter(P_db, size=31)                     # glattes Grundrauschen
    toene = np.maximum(P_db - grund, 0)                      # nur was als Ton herausragt
    bis_2k = toene[f < 2000]
    baender = bis_2k[: len(bis_2k) // 8 * 8].reshape(-1, 8).max(axis=1)   # 64 Baender je 31 Hz
    flach = np.exp(np.mean(np.log(P + EPS))) / (np.mean(P) + EPS)         # 1 = Rauschen, 0 = Ton
    return np.append(baender, flach)


def huelle(y):
    """Rhythmus der Lautstaerke von 2 bis 200 Hz in Schritten von 2 Hz. 99 Werte."""
    lautstaerke = np.abs(hilbert(y))
    lautstaerke = lautstaerke - lautstaerke.mean()
    spektrum = np.abs(np.fft.rfft(lautstaerke * np.hanning(len(lautstaerke))))
    f = np.fft.rfftfreq(len(lautstaerke), 1 / SR)
    werte = [spektrum[(f >= a) & (f < a + 2)].mean() for a in range(2, 200, 2)]
    return np.log10(np.array(werte) + EPS)


MERKMALE = {
    "ton": ton,
    "huelle": huelle,
    "kombi": lambda y: np.concatenate([mel_mittel(y), ton(y), huelle(y)]),
}


def auc(X_train, X_test, label):
    """Skalierung und Isolation Forest wie auf der Box, AUC gemittelt ueber 5 Startwerte."""
    skala = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    X_train, X_test = skala.transform(X_train), skala.transform(X_test)
    werte = []
    for seed in range(5):
        wald = IsolationForest(contamination=0.01, random_state=seed).fit(X_train)
        werte.append(roc_auc_score(label, -wald.score_samples(X_test)))
    return np.mean(werte)


zeilen = [["maschine", "verfahren", *MERKMALE]]

for maschine in sorted(DATEN.iterdir()):
    normal = sorted((maschine / "normal").glob("*.wav"))
    fehler = sorted((maschine / "abnormal").glob("*.wav"))

    # Test: alle Fehler und gleich viele Normalaufnahmen, Rest ist Training
    train = normal[len(fehler):]
    test = normal[:len(fehler)] + fehler
    label = [0] * len(fehler) + [1] * len(fehler)

    for verfahren, verarbeiten in [("roh", lambda y: y),
                                   ("getrennt", lambda y: spektralsubtraktion(y, SR)[0])]:
        Y_train = [verarbeiten(laden(d)) for d in train]
        Y_test = [verarbeiten(laden(d)) for d in test]

        werte = []
        for f in MERKMALE.values():
            X_train = np.array([f(y) for y in Y_train])
            X_test = np.array([f(y) for y in Y_test])
            werte.append(round(auc(X_train, X_test, label), 3))

        zeilen.append([maschine.name, verfahren, *werte])
        print(f"{maschine.name}  {verfahren:9s}  " +
              "   ".join(f"{n} {w:.3f}" for n, w in zip(MERKMALE, werte)))

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / "mimii_physik.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
