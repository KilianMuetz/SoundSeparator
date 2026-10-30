"""
engine_trennstufe.py — Fehlalarme durch Stoerschall, roh und nach der Trennung.

Datensatz IDMT-ISA-ELECTRIC-ENGINE (Grollmisch et al. 2019): Elektromotor,
Stoerschall live waehrend der Aufnahme: Sprechen, Klopfen aufs Gehaeuse,
Hallenatmosphaere, weisses Rauschen.

Gut und defekt sind verschiedene Motoren (engine1, engine2). Gut gegen defekt
misst deshalb auch den Motor, nicht nur den Fehler. Hauptmass sind daher die
Fehlalarme am guten Motor: trainiert auf seiner sauberen Aufnahme, getestet
mit Stoerschall. Jeder Alarm dort ist ein Fehlalarm.
Kontrolle: Treffer am defekten Motor unter denselben Bedingungen.

Training: erste 80 % von train/engine1_good/pure.wav
Schwelle: 90. Perzentil der Scores der letzten 20 % (wie Dohi et al. 2022)
Segmente: 5 s wie SIPREMA, Versatz 2,5 s, auf 16 kHz umgerechnet.
Merkmale und Detektor wie auf der Box: 20 MFCC, Hoechstwert, Skalierung, Isolation Forest.
Ergebnis: ergebnisse/engine_trennstufe.csv
"""

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from trennung import spektralsubtraktion

BASE = Path(__file__).parent.parent
DATEN = BASE / "data" / "engine"
ERGEBNISSE = BASE / "ergebnisse"
BOX = BASE.parent / "siprema" / "box" / "backend"
SR = 16000
DAUER = 5 * SR
VERSATZ = DAUER // 2

# Parameter der MFCC direkt aus dem Box-Code
sys.path.insert(0, str(BOX))
from src import feature_schema as fs  # noqa: E402

BEDINGUNGEN = {
    "sprechen": ["talking_1", "talking_2", "talking_3", "talking_4"],
    "klopfen": ["stresstest"],
    "atmo_leise": ["atmo_low"],
    "atmo_mittel": ["atmo_medium"],
    "atmo_laut": ["atmo_high"],
    "rauschen": ["whitenoise_low"],
}


def segmente(datei):
    """Ganze Aufnahme auf 16 kHz, zerlegt in 5-s-Segmente mit 2,5 s Versatz."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True)
    y = resample_poly(y[:, 0], SR, sr)
    return [y[s:s + DAUER] for s in range(0, len(y) - DAUER + 1, VERSATZ)]


def merkmale(y):
    """MFCC-Hoechstwert wie auf der Box. Die Box rechnet mit 16-bit-Werten."""
    x = np.clip(y * 32767, -32768, 32767).astype(np.int16).astype(np.float64)
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=fs.FRAME_SIZE, hop_length=fs.HOP_SIZE,
                                         n_mels=fs.N_MELS, fmin=fs.MEL_FMIN, norm=fs.MEL_NORM)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, top_db=fs.MFCC_TOP_DB),
                                n_mfcc=fs.N_MFCC, dct_type=fs.MFCC_DCT_TYPE, norm=fs.MFCC_NORM)
    return mfcc.max(axis=1)


def alarmquote(Y_train, Y_kalib, Y_test):
    """Anteil der Testsegmente ueber der Schwelle, gemittelt ueber 5 Startwerte."""
    skala = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(Y_train)
    quoten = []
    for seed in range(5):
        wald = IsolationForest(contamination=0.01, random_state=seed).fit(skala.transform(Y_train))
        score = lambda X: -wald.score_samples(skala.transform(X))
        schwelle = np.percentile(score(Y_kalib), 90)
        quoten.append(np.mean(score(Y_test) > schwelle))
    return 100 * np.mean(quoten)


zeilen = [["bedingung", "verfahren", "fehlalarme_gut_prozent", "treffer_defekt_prozent"]]

for verfahren, verarbeiten in [("roh", lambda y: y),
                               ("speksub", lambda y: spektralsubtraktion(y, SR)[0])]:
    rein = segmente(DATEN / "train" / "engine1_good" / "pure.wav")
    grenze = int(0.8 * len(rein))
    X_train = np.array([merkmale(verarbeiten(y)) for y in rein[:grenze]])
    X_kalib = np.array([merkmale(verarbeiten(y)) for y in rein[grenze:]])

    for bedingung, dateien in BEDINGUNGEN.items():
        X = {}
        for motor in ["engine1_good", "engine2_broken"]:
            segs = [s for d in dateien for s in segmente(DATEN / "test" / motor / f"{d}.wav")]
            X[motor] = np.array([merkmale(verarbeiten(y)) for y in segs])

        fehlalarme = alarmquote(X_train, X_kalib, X["engine1_good"])
        treffer = alarmquote(X_train, X_kalib, X["engine2_broken"])
        zeilen.append([bedingung, verfahren, round(fehlalarme, 1), round(treffer, 1)])
        print(f"{bedingung:12s} {verfahren:8s}  Fehlalarme gut {fehlalarme:5.1f} %   "
              f"Treffer defekt {treffer:5.1f} %")

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / "engine_trennstufe.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
