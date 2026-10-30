"""
mischtest.py — Fehlalarme durch kurze Stoerereignisse, roh und nach der Trennung.

Basis: MIMII-Aufnahmen (Maschine und gleichmaessiger Hallenlaerm, eine Sitzung).
Dazu gemischt: kurze Ereignisse aus der Masterarbeit (Tuerschlaege, Metallhammer,
Gespraech), ueber die vollen 5 s, gleich laut wie die Aufnahme (0 dB).
Anders als in der Masterarbeit gibt es keinen stoerungsfreien Vorlauf.

Training: Normalaufnahmen ohne Ereignisse.
Schwelle: 90. Perzentil auf ungesehenen Normalaufnahmen (wie Dohi et al. 2022).
Test je Ereignis:
  Fehlalarme = Anteil Normalaufnahmen + Ereignis ueber der Schwelle (niedriger ist besser)
  Treffer    = Anteil Fehleraufnahmen + Ereignis ueber der Schwelle (hoeher ist besser)
Merkmale und Detektor wie auf der Box: 20 MFCC, Hoechstwert, Skalierung, Isolation Forest.
Ergebnis: ergebnisse/mischtest_<maschine>.csv
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

MASCHINE = "pump"   # "fan" oder "pump"

BASE = Path(__file__).parent.parent
DATEN = BASE / "data" / MASCHINE
STOER = BASE.parent / "SoundSeparator" / "data" / "set"
ERGEBNISSE = BASE / "ergebnisse"
BOX = BASE.parent / "siprema" / "box" / "backend"
SR = 16000
DAUER = 5 * SR
N_TEST = 100      # Normal- und Fehleraufnahmen je Maschine im Test
N_KALIB = 100     # Normalaufnahmen fuer die Schwelle

EREIGNISSE = {
    "tuerschlaege": "tuerschlaege.wav",
    "metallhammer": "metallhammer.wav",
    "gespraech": "gespraech_6dzb.wav",
}

# Parameter der MFCC direkt aus dem Box-Code
sys.path.insert(0, str(BOX))
from src import feature_schema as fs  # noqa: E402


def laden(datei):
    """Kanal 1, erste 5 s."""
    y, sr = sf.read(datei, dtype="float64", always_2d=True, frames=DAUER)
    assert sr == SR, f"{datei}: Abtastrate {sr}"
    return y[:, 0]


def stoerquelle(datei):
    """Ganze Stoeraufnahme, Mono, 16 kHz."""
    y, sr = sf.read(STOER / datei, dtype="float64", always_2d=True)
    return resample_poly(y.mean(axis=1), SR, sr)


def mischen(y, stoer, zufall):
    """5-s-Ausschnitt der Stoerquelle an zufaelliger Stelle, gleich laut wie y."""
    start = zufall.integers(0, len(stoer) - DAUER)
    s = stoer[start:start + DAUER]
    rms = lambda x: np.sqrt(np.mean(x ** 2)) + 1e-12
    return y + s * rms(y) / rms(s)


def merkmale(y):
    """MFCC-Hoechstwert wie auf der Box. Die Box rechnet mit 16-bit-Werten."""
    x = np.clip(y * 32767, -32768, 32767).astype(np.int16).astype(np.float64)
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=fs.FRAME_SIZE, hop_length=fs.HOP_SIZE,
                                         n_mels=fs.N_MELS, fmin=fs.MEL_FMIN, norm=fs.MEL_NORM)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, top_db=fs.MFCC_TOP_DB),
                                n_mfcc=fs.N_MFCC, dct_type=fs.MFCC_DCT_TYPE, norm=fs.MFCC_NORM)
    return mfcc.max(axis=1)


def alarmquote(X_train, X_kalib, X_test):
    """Anteil der Testsegmente ueber der Schwelle in %, gemittelt ueber 5 Startwerte."""
    skala = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train)
    quoten = []
    for seed in range(5):
        wald = IsolationForest(contamination=0.01, random_state=seed).fit(skala.transform(X_train))
        score = lambda X: -wald.score_samples(skala.transform(X))
        schwelle = np.percentile(score(X_kalib), 90)
        quoten.append(np.mean(score(X_test) > schwelle))
    return 100 * np.mean(quoten)


stoer = {name: stoerquelle(datei) for name, datei in EREIGNISSE.items()}
verfahren = {"roh": lambda y: y, "speksub": lambda y: spektralsubtraktion(y, SR)[0]}
ergebnis = {}   # (ereignis, verfahren) -> Liste [fehlalarme, treffer] je Maschine

for maschine in sorted(DATEN.iterdir()):
    normal = sorted((maschine / "normal").glob("*.wav"))
    fehler = sorted((maschine / "abnormal").glob("*.wav"))[:N_TEST]
    test_normal = normal[:N_TEST]
    kalib = normal[N_TEST:N_TEST + N_KALIB]
    train = normal[N_TEST + N_KALIB:]

    Y_test_normal = [laden(d) for d in test_normal]
    Y_fehler = [laden(d) for d in fehler]

    for v_name, v in verfahren.items():
        X_train = np.array([merkmale(v(laden(d))) for d in train])
        X_kalib = np.array([merkmale(v(laden(d))) for d in kalib])

        for e_name in ["ohne", *EREIGNISSE]:
            zufall = np.random.default_rng(0)   # gleiche Ausschnitte fuer roh und speksub
            mix = (lambda y: y) if e_name == "ohne" else (lambda y: mischen(y, stoer[e_name], zufall))
            X_n = np.array([merkmale(v(mix(y))) for y in Y_test_normal])
            X_f = np.array([merkmale(v(mix(y))) for y in Y_fehler])
            werte = [alarmquote(X_train, X_kalib, X_n), alarmquote(X_train, X_kalib, X_f)]
            ergebnis.setdefault((e_name, v_name), []).append(werte)
            print(f"{maschine.name}  {e_name:13s} {v_name:8s}  "
                  f"Fehlalarme {werte[0]:5.1f} %   Treffer {werte[1]:5.1f} %")

print(f"\nMittel ueber alle Maschinen ({MASCHINE})")
zeilen = [["ereignis", "verfahren", "fehlalarme_prozent", "treffer_prozent"]]
for (e_name, v_name), werte in ergebnis.items():
    fa, tr = np.mean(werte, axis=0)
    zeilen.append([e_name, v_name, round(fa, 1), round(tr, 1)])
    print(f"{e_name:13s} {v_name:8s}  Fehlalarme {fa:5.1f} %   Treffer {tr:5.1f} %")

ERGEBNISSE.mkdir(exist_ok=True)
with open(ERGEBNISSE / f"mischtest_{MASCHINE}.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(zeilen)
