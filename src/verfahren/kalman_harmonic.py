from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/kalman_harmonic/kalman_harmonicNs.wav"   # harmonischer Anteil
ausgabe_hs = BASE / "../sep/kalman_harmonic/kalman_harmonicHs.wav"   # Residuum

f_r = 25.0        # Grundfrequenz der Rotation in Hz (= rpm/60)
H = 20            # Anzahl getrackter Harmonischer
q = 1e-4          # Prozessrauschen (klein = starres harmonisches Modell)
r = 1.0           # Messrauschen (gross = mehr geht ins Residuum/Stoerschall)

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Zustandsuebergang A: blockdiagonale Rotationsmatrizen je Harmonische ---
dim = 2 * H
A = np.zeros((dim, dim))
for h in range(H):
    w = 2 * np.pi * (h + 1) * f_r / sr
    c, s = np.cos(w), np.sin(w)
    A[2*h:2*h+2, 2*h:2*h+2] = [[c, -s], [s, c]]

# --- Beobachtungsvektor C: Summe der cos-Komponenten = Signalwert ---
C = np.zeros(dim)
C[0::2] = 1.0

# --- Kalman-Rekursion ---
Q = q * np.eye(dim)
x = np.zeros(dim)
P = np.eye(dim)
ns = np.zeros(N)

for k in range(N):
    # Vorhersage
    x = A @ x
    P = A @ P @ A.T + Q
    # Korrektur
    y_pred = C @ x
    S = C @ P @ C + r
    Kk = (P @ C) / S
    x = x + Kk * (y[k] - y_pred)
    P = P - np.outer(Kk, C) @ P
    ns[k] = C @ x

# --- Residuum = Stoerschall ---
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"{H} Harmonische getrackt (f_r = {f_r} Hz)")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")