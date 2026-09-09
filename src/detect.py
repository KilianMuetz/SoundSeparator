"""
detect.py — Merkmalsextraktion und unueberwachte Anomalieerkennung.

Je Trennverfahren wird ein eigener Isolation Forest ausschliesslich auf
Normaldaten trainiert, die zuvor durch dasselbe Verfahren verarbeitet wurden.
Damit lernt jedes Modell die Artefakte seines Verfahrens als normal und
bewertet nicht die Trennung, sondern die Maschinenanomalie.

Zusaetzlich wird das unbearbeitete Mischsignal als Referenz ausgewertet
("roh"). Der Vergleich gegen diese Referenz zeigt, ob die Trennung die
Anomalieerkennung ueberhaupt verbessert.

Merkmale: 20 MFCCs je Frame (25 ms Fenster, 10 ms Versatz).
Bewertung: mittlerer Anomaliescore ueber alle Frames eines Segments.

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
SEED = 0


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


def score(modell, skalierer, X):
    """Anomaliescore: je hoeher, desto anomaler."""
    return float(np.mean(-modell.score_samples(skalierer.transform(X))))


# --- Mischsignale ---
with open(mix_dir / "manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

ist_normal = [e["nutzschall"] == "normal" for e in manifest]
labels = [0 if n else 1 for n in ist_normal]
normal_idx = [i for i, n in enumerate(ist_normal) if n]

merkmal_dir.mkdir(parents=True, exist_ok=True)
protokoll_dir.mkdir(parents=True, exist_ok=True)

zeilen = []
zusammenfassung = []

for verfahren in ["roh"] + list(vl.VERFAHREN):
    # --- Merkmale aller 84 Segmente ---
    X = [merkmale(signal_laden(verfahren, e)) for e in manifest]
    np.savez_compressed(merkmal_dir / f"{verfahren}.npz",
                        **{e["mix_id"]: x for e, x in zip(manifest, X)})

    scores = np.zeros(len(manifest))

    # --- Modell auf allen Normaldaten: bewertet die Anomaliesegmente ---
    X_train = np.vstack([X[i] for i in normal_idx])
    skalierer = StandardScaler().fit(X_train)
    modell = IsolationForest(random_state=SEED, n_estimators=100)
    modell.fit(skalierer.transform(X_train))

    for i, x in enumerate(X):
        if not ist_normal[i]:
            scores[i] = score(modell, skalierer, x)

    # --- Normaldaten: Leave-one-out, damit ihr Score nicht im Training steckt ---
    for i in normal_idx:
        rest = [j for j in normal_idx if j != i]
        X_r = np.vstack([X[j] for j in rest])
        sk_r = StandardScaler().fit(X_r)
        mo_r = IsolationForest(random_state=SEED, n_estimators=100)
        mo_r.fit(sk_r.transform(X_r))
        scores[i] = score(mo_r, sk_r, X[i])

    auc = roc_auc_score(labels, scores)
    zusammenfassung.append({"verfahren": verfahren, "auc": round(auc, 4)})
    print(f"{verfahren:24s} AUC = {auc:.4f}")

    for e, s, lab in zip(manifest, scores, labels):
        zeilen.append({
            "mix_id": e["mix_id"],
            "nutzschall": e["nutzschall"],
            "stoerquelle": e["stoerquelle"],
            "verfahren": verfahren,
            "label": lab,
            "score": round(float(s), 6),
        })

# --- Protokolle schreiben ---
with open(protokoll_dir / "erkennung.csv", "w", newline="", encoding="utf-8") as f:
    schreiber = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()))
    schreiber.writeheader()
    schreiber.writerows(zeilen)

with open(protokoll_dir / "erkennung_auc.csv", "w", newline="", encoding="utf-8") as f:
    schreiber = csv.DictWriter(f, fieldnames=["verfahren", "auc"])
    schreiber.writeheader()
    schreiber.writerows(zusammenfassung)

print(f"\nScores:  {protokoll_dir / 'erkennung.csv'}")
print(f"AUC:     {protokoll_dir / 'erkennung_auc.csv'}")