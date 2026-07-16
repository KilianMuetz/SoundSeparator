from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/statespace_em/statespace_emNs.wav"
ausgabe_hs = BASE / "../sep/statespace_em/statespace_emHs.wav"

dim = 20          # Zustandsdimension (AR-artige Modellordnung)
n_em = 8          # Anzahl EM-Iterationen
nperseg = 1024    # (nur fuer evtl. Analyse; hier zeitbereichsbasiert)

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Feste Modellstruktur: Beobachtung pickt erste Zustandskomponente ---
C = np.zeros(dim)
C[0] = 1.0

# --- Initialisierung der Parameter ---
A = np.eye(dim) * 0.9
A[1:, :-1] = np.eye(dim - 1)         # Begleit-/Verschiebungsstruktur
A[0, :] = 0.5 / dim
Q = np.eye(dim) * np.var(y) * 0.1
R = np.var(y) * 0.5
x0 = np.zeros(dim)
P0 = np.eye(dim) * np.var(y)

# --- Kalman-Filter + RTS-Smoother (E-Schritt) ---
def kalman_rts(y, A, C, Q, R, x0, P0):
    N = len(y)
    d = A.shape[0]
    xf = np.zeros((N, d)); Pf = np.zeros((N, d, d))
    xp = np.zeros((N, d)); Pp = np.zeros((N, d, d))
    x = x0.copy(); P = P0.copy()
    for k in range(N):
        x = A @ x; P = A @ P @ A.T + Q
        xp[k] = x; Pp[k] = P
        yp = C @ x; S = C @ P @ C + R; K = (P @ C) / S
        x = x + K * (y[k] - yp); P = P - np.outer(K, C) @ P
        xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy(); Js = np.zeros((N, d, d))
    for k in range(N - 2, -1, -1):
        J = Pf[k] @ A.T @ np.linalg.inv(Pp[k + 1] + 1e-9 * np.eye(d))
        xs[k] = xf[k] + J @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + J @ (Ps[k + 1] - Pp[k + 1]) @ J.T
        Js[k] = J
    return xs, Ps, Js

# --- EM-Schleife ---
for it in range(n_em):
    xs, Ps, Js = kalman_rts(y, A, C, Q, R, x0, P0)
    # Erwartungswert-Summen fuer den M-Schritt
    S11 = np.zeros((dim, dim)); S00 = np.zeros((dim, dim)); S10 = np.zeros((dim, dim))
    for k in range(1, N):
        S11 += np.outer(xs[k], xs[k]) + Ps[k]
        S00 += np.outer(xs[k - 1], xs[k - 1]) + Ps[k - 1]
        S10 += np.outer(xs[k], xs[k - 1]) + Js[k - 1] @ Ps[k]
    # M-Schritt: erste Zeile von A lernen (AR-Struktur), Q und R aktualisieren
    A_new = S10 @ np.linalg.inv(S00 + 1e-9 * np.eye(dim))
    A[0, :] = A_new[0, :]
    Q[0, 0] = max((S11[0, 0] - A[0, :] @ S10[0, :]) / N, 1e-8)
    acc = 0.0
    for k in range(N):
        acc += (y[k] - C @ xs[k]) ** 2 + C @ Ps[k] @ C
    R = max(acc / N, 1e-8)

# --- Finaler E-Schritt liefert den Nutzschall ---
xs, _, _ = kalman_rts(y, A, C, Q, R, x0, P0)
ns = xs[:, 0]
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"dim = {dim}, {n_em} EM-Iterationen")
print(f"gelernt: q = {Q[0,0]:.2e}, r = {R:.2e}")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")