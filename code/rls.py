"""
RLS-ANC - Adaptive Noise Cancelling mit Recursive Least Squares
Wie das LMS-basierte ANC nutzt dieses Verfahren eine synthetische, drehzahl-
synchrone Referenz ([cos, sin] fuer f_r und Harmonische) und rekonstruiert
adaptiv den drehzahlsynchronen Anteil. Statt des einfachen LMS-Gradienten-
schritts wird der Filter per Recursive Least Squares aktualisiert: eine
rekursive Matrixgleichung mit inverser Korrelationsmatrix P und Vergessens-
faktor lambda minimiert den kumulierten quadratischen Fehler.

RLS vs. LMS (der klassische Vergleich, vgl. Haykin, Adaptive Filter Theory):
  + RLS konvergiert deutlich schneller und ist unempfindlich gegen die
    Schrittweiten-Wahl, die LMS erschwert.
  - RLS kostet O(M^2) pro Abtastwert statt O(M) und ist numerisch
    empfindlicher: bei stark korrelierter (rein harmonischer) Referenz muss
    lambda nahe 1 liegen (>= 0.9999), sonst divergiert die Rekursion.

Hinweis zum Verhalten: Durch die sehr schnelle Anpassung legt RLS die Referenz
nahezu momentanoptimal an; der als Nutzschall rekonstruierte Anteil faellt
dadurch kleiner aus als bei LMS, waehrend Transienten vollstaendig im Fehler
landen. Dieser Trade-off (schnelle Konvergenz vs. Selektivitaet/Rechenkosten)
ist der eigentliche Erkenntniswert gegenueber dem LMS-ANC.
  Ns = adaptiv rekonstruierter drehzahlsynchroner Anteil   = Nutzschall
  Hs = Fehlersignal y - Ns                                 = Stoerschall
"""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/rls/rlsNS.wav"   # drehzahlsynchron
ausgabe_hs = BASE / "../sep/rls/rlsHs.wav"   # Fehlersignal

f_r = 48.0        # Grundfrequenz der Rotation in Hz
H = 40            # Anzahl Harmonische in der Referenz
lam = 0.9999      # Vergessensfaktor (muss nahe 1 sein, sonst instabil)
delta = 1.0       # Init der inversen Korrelationsmatrix P = delta * I

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Synthetische harmonische Referenz: cos/sin je Harmonische ---
M = 2 * H
n = np.arange(N)
ref = np.zeros((M, N))
for h in range(H):
    w_h = 2 * np.pi * (h + 1) * f_r / sr
    ref[2*h]     = np.cos(w_h * n)
    ref[2*h + 1] = np.sin(w_h * n)

# --- RLS-Rekursion ---
w = np.zeros(M)
P = np.eye(M) * delta          # inverse Korrelationsmatrix
ns = np.zeros(N)
for k in range(N):
    x = ref[:, k]
    Px = P @ x
    g = Px / (lam + x @ Px)     # Gain-Vektor
    e = y[k] - w @ x            # a-priori-Fehler
    w = w + g * e              # Gewichts-Update
    P = (P - np.outer(g, Px)) / lam
    ns[k] = w @ x              # Rekonstruktion mit aktualisierten Gewichten

# --- Fehlersignal = Stoerschall ---
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"f_r = {f_r} Hz, {H} Harmonische, lambda = {lam}")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")