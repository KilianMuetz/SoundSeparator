from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/ssa/ssaNs.wav"   # stationaer
ausgabe_hs = BASE / "../sep/ssa/ssaHs.wav"   # Residuum

L = 300           # Fensterlaenge (Einbettungsdimension)
r = 5             # Anzahl fuehrender Komponenten -> Nutzschall
seg_len = 8000    # Blocklaenge fuer die segmentweise Verarbeitung

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Diagonale Mittelung (Hankelisierung) einer Matrix zurueck ins Zeitsignal ---
def hankelize(M):
    Lr, Kr = M.shape
    out = np.zeros(Lr + Kr - 1)
    cnt = np.zeros(Lr + Kr - 1)
    for i in range(Lr):
        out[i:i + Kr] += M[i]
        cnt[i:i + Kr] += 1
    return out / cnt

# --- SSA auf einem Segment: r fuehrende Komponenten rekonstruieren ---
def ssa_segment(seg, L, r):
    Ns_seg = len(seg)
    K = Ns_seg - L + 1
    X = np.column_stack([seg[i:i + L] for i in range(K)])   # Trajektorienmatrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_r = (U[:, :r] * S[:r]) @ Vt[:r]                        # fuehrende r Komponenten
    return hankelize(X_r)

# --- Blockweise ueber das ganze Signal ---
ns = np.zeros(N)
for start in range(0, N, seg_len):
    seg = y[start:start + seg_len]
    if len(seg) < L + 10:                  # Restblock zu kurz -> unveraendert
        ns[start:start + len(seg)] = seg
    else:
        ns[start:start + len(seg)] = ssa_segment(seg, L, r)

# --- Residuum = Stoerschall ---
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"L = {L}, r = {r}, Blocklaenge {seg_len}")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")