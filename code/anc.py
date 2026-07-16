"""
ANC - Adaptive Noise Cancelling (Widrow et al., 1975), Single-Channel-Variante
Klassisches ANC braucht einen Referenzkanal mit korrelierter Stoerung. Da das
SIPREMA-Setup effektiv einkanalig ist, wird eine SYNTHETISCHE, drehzahl-
synchrone Referenz erzeugt: [cos, sin]-Paare fuer f_r und dessen Harmonische.
Ein adaptiver LMS-Filter lernt online, wie diese Referenz im gemessenen Signal
erscheint (Amplitude/Phase je Harmonische), und rekonstruiert daraus den
drehzahlsynchronen Anteil. Der Fehler (Signal minus Rekonstruktion) enthaelt
alles Nicht-Synchrone.
  Ns = adaptiv rekonstruierter drehzahlsynchroner Anteil   = Nutzschall
  Hs = Fehlersignal y - Ns                                 = Stoerschall

Der adaptive Filter passt sich langsamen Amplituden-/Phasenaenderungen an
(im Gegensatz zum starren Kammfilter). Transiente Stoerungen (Schlaege) passen
nicht zur harmonischen Referenz und erscheinen vollstaendig im Fehlersignal.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/anc/ancNs.wav"   # drehzahlsynchron
ausgabe_hs = BASE / "../sep/anc/ancHs.wav"   # Fehlersignal

f_r = 48.0        # Grundfrequenz der Rotation in Hz
H = 40            # Anzahl Harmonische in der Referenz
mu = 0.1          # LMS-Schrittweite (gross = schnell, aber Overfitting-Gefahr)
leak = 1e-5       # Leakage (verhindert Gewichts-Wegdriften, erhoeht Robustheit)

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

# --- Normalisiertes LMS mit Leakage ---
w = np.zeros(M)
ns = np.zeros(N)
for k in range(N):
    rk = ref[:, k]
    yhat = w @ rk                       # aktuelle Rekonstruktion
    e = y[k] - yhat                     # Fehler = Stoerschall-Anteil
    w = (1 - leak) * w + mu * e * rk / (rk @ rk + 1e-6)
    ns[k] = yhat

# --- Fehlersignal = Stoerschall ---
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"f_r = {f_r} Hz, {H} Harmonische, mu = {mu}")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")