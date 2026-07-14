from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/kalman_ar/ArKalmanNs.wav"   # AR-modelliert
ausgabe_hs = BASE / "../sep/kalman_ar/ArKalmanHs.wav"   # Residuum

p = 20                    # AR-Ordnung (hoeher = feinere Spektralstruktur)
ref_ende_s = 1.0          # ruhiges Referenzsegment (nur Nutzschall) bis hier
q = 1e-3                  # Prozessrauschen
r_meas = 0.1              # Messrauschen (gross = mehr geht ins Residuum)

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- AR-Koeffizienten per Levinson-Durbin aus dem Referenzsegment ---
ref = y[:int(ref_ende_s * sr)]
rk = np.correlate(ref, ref, mode="full")[len(ref) - 1:len(ref) + p]
a = np.zeros(p)
err = rk[0]
for i in range(p):
    acc = rk[i + 1] - np.dot(a[:i], rk[1:i + 1][::-1])
    k = acc / (err + 1e-12)
    a_new = a.copy()
    a_new[i] = k
    for j in range(i):
        a_new[j] = a[j] - k * a[i - 1 - j]
    a = a_new
    err = err * (1 - k ** 2)

# --- Kalman mit AR-Prozessmodell (Begleitmatrix) ---
dim = p
A = np.zeros((dim, dim))
A[0, :] = a                       # AR-Vorhersage in erster Zeile
A[1:, :-1] = np.eye(dim - 1)      # Verschiebung der Zustandshistorie
C = np.zeros(dim)
C[0] = 1.0
Q = np.zeros((dim, dim))
Q[0, 0] = q

x = np.zeros(dim)
P = np.eye(dim)
ns = np.zeros(N)

for k in range(N):
    # Vorhersage
    x = A @ x
    P = A @ P @ A.T + Q
    # Korrektur
    y_pred = C @ x
    S = C @ P @ C + r_meas
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
print(f"AR-Ordnung p = {p}, Referenz bis {ref_ende_s}s")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")