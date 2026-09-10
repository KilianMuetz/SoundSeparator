"""
detect.py — Merkmalsextraktion und unueberwachte Anomalieerkennung.

Je Trennverfahren wird ein eigener Isolation Forest ausschliesslich auf
Normaldaten trainiert, die zuvor durch dasselbe Verfahren verarbeitet wurden.
Damit lernt jedes Modell die Artefakte seines Verfahrens als normal und
bewertet nicht die Trennung, sondern die Maschinenanomalie.

Trainings- und Testnormaldaten stammen aus zeitlich getrennten Abschnitten
der Normalaufnahme (Feld "rolle" im Manifest). Damit kann kein Testsegment
aus seinem eigenen Trainingsmaterial bewertet werden.

Zusaetzlich wird das unbearbeitete Mischsignal als Referenz ausgewertet
("roh"). Der Vergleich gegen diese Referenz zeigt, ob die Trennung die
Anomalieerkennung ueberhaupt verbessert.

Merkmale: 20 MFCCs je Frame (25 ms Fenster, 10 ms Versatz).
Bewertung: mittlerer Anomaliescore ueber alle Frames eines Segments,
gemittelt ueber mehrere Isolation-Forest-Initialisierungen.

Aufruf:  python src/detect.py
"""

import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import verfahren_lib as vl

# --- Pfade ---
BASE = Path(__file__).parent.parent
mix_dir = BASE / "data" / "mixed"
getrennt_dir = BASE / "ergebnisse" / "getrennt"
merkmal_dir = BASE / "ergebnisse" / "merkmale"
protokoll_dir = BASE / "ergebnisse" / "protokoll"

# --- Parameter ---
SR = 16000
N_MFCC = 20        # in Abstimmung mit Roewaplan
N_FFT = 512
WIN_LENGTH = 400   # 25 ms
HOP_LENGTH = 160   # 10 ms
SEEDS = [0, 1, 2, 3, 4]   # Mittelung ueber mehrere Isolation-Forest-Initialisierungen


def merkmale(y):
    """20 MFCCs je Frame, Ergebnis (n_frames, N_MFCC)."""
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
                             win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    return m.T.astype(np.float64)


def signal_laden(verfahren, eintrag):
    """Getrenntes Nutzsignal, fuer 'roh' das dezimierte Mischsignal."""
    if verfahren == "roh":
        y, sr = sf.read(mix_dir / eintrag["file"], dtype="float64")
        return resample_poly(y, SR // 100, sr // 100)
    y, _ = sf.read(getrennt_dir / verfahren / f"{eintrag['mix_id']}_Ns.wav",
                   dtype="float64")
    return y


# --- Mischsignale ---
with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

train_idx = [i for i, e in enumerate(manifest) if e["rolle"] == "train"]
test_idx = [i for i, e in enumerate(manifest) if e["rolle"] == "test"]
labels = [0 if e["nutzschall"] == "normal" else 1 for e in manifest]

assert all(labels[i] == 0 for i in train_idx), "Trainingsmenge enthaelt Anomalien"
print(f"{len(train_idx)} Trainingssegmente, {len(test_idx)} Testsegmente "
      f"(davon {sum(1 for i in test_idx if labels[i] == 0)} normal, "
      f"{sum(labels[i] for i in test_idx)} anomal)\n")

merkmal_dir.mkdir(parents=True, exist_ok=True)
protokoll_dir.mkdir(parents=True, exist_ok=True)

zeilen = []
zusammenfassung = []

for verfahren in ["roh"] + list(vl.VERFAHREN):
    # --- Merkmale aller Segmente ---
    X = [merkmale(signal_laden(verfahren, e)) for e in manifest]
    np.savez_compressed(merkmal_dir / f"{verfahren}.npz",
                        **{e["mix_id"]: x for e, x in zip(manifest, X)})

    # --- Training auf den Normaldaten des Trainingsbereichs, ueber Seeds gemittelt ---
    X_train = np.vstack([X[i] for i in train_idx])
    skalierer = StandardScaler().fit(X_train)
    X_train_s = skalierer.transform(X_train)

    scores_je_seed = []
    aucs = []
    for seed in SEEDS:
        modell = IsolationForest(random_state=seed, n_estimators=100)
        modell.fit(X_train_s)
        s = {i: float(np.mean(-modell.score_samples(skalierer.transform(X[i]))))
             for i in test_idx}
        scores_je_seed.append(s)
        aucs.append(roc_auc_score([labels[i] for i in test_idx],
                                  [s[i] for i in test_idx]))

    scores = {i: float(np.mean([s[i] for s in scores_je_seed])) for i in test_idx}

    # --- Diagnose: kollabierte Normalscores deuten auf ein Datenleck hin ---
    norm = np.array([scores[i] for i in test_idx if labels[i] == 0])
    anom = np.array([scores[i] for i in test_idx if labels[i] == 1])

    zusammenfassung.append({
        "verfahren": verfahren,
        "auc": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "score_normal": round(float(norm.mean()), 4),
        "streuung_normal": round(float(norm.std()), 4),
        "score_anomal": round(float(anom.mean()), 4),
        "streuung_anomal": round(float(anom.std()), 4),
    })
    print(f"{verfahren:24s} AUC = {np.mean(aucs):.4f} +- {np.std(aucs):.4f}   "
          f"Streuung normal {norm.std():.4f}")

    for i in test_idx:
        e = manifest[i]
        zeilen.append({
            "mix_id": e["mix_id"],
            "nutzschall": e["nutzschall"],
            "stoerquelle": e["stoerquelle"],
            "offset_s": e["offset_s"],
            "verfahren": verfahren,
            "label": labels[i],
            "score": round(scores[i], 6),
        })

# --- Protokolle schreiben ---
with open(protokoll_dir / "erkennung.csv", "w", newline="", encoding="utf-8") as f:
    schreiber = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()))
    schreiber.writeheader()
    schreiber.writerows(zeilen)

with open(protokoll_dir / "erkennung_auc.csv", "w", newline="", encoding="utf-8") as f:
    schreiber = csv.DictWriter(f, fieldnames=list(zusammenfassung[0].keys()))
    schreiber.writeheader()
    schreiber.writerows(zusammenfassung)

print(f"\nScores:  {protokoll_dir / 'erkennung.csv'}")
print(f"AUC:     {protokoll_dir / 'erkennung_auc.csv'}")
