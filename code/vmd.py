from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/vmd/VmdNs.wav"
ausgabe_hs = BASE / "../sep/vmd/VmdHs.wav"

f_r = 25.0        # Rotationsgrundfrequenz in Hz (= rpm/60), fuer die Zuordnung
K = 6             # Anzahl der Moden
alpha = 2000.0    # Bandbreiten-Straffheit (gross = schmalere Moden)
tau = 0.0         # Rauschtoleranz des Lagrange-Multiplikators (0 = strikt)
n_iter = 500      # max. ADMM-Iterationen
tol = 1e-7        # Konvergenzschwelle

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Spiegelung an den Raendern (reduziert Randartefakte) ---
f_mir = np.concatenate([y[N // 2:0:-1], y, y[-1:-N // 2 - 1:-1]])
T = len(f_mir)
freqs = np.arange(T) / T - 0.5                  # normierte Frequenz [-0.5, 0.5)

f_hat = np.fft.fftshift(np.fft.fft(f_mir))
f_hat_plus = f_hat.copy()
f_hat_plus[:T // 2] = 0                          # nur positives Halbband

# --- Mittenfrequenzen an den K groessten Spektralpeaks initialisieren ---
mag = np.abs(f_hat_plus[T // 2:])
idx, _ = find_peaks(mag, height=mag.max() * 0.05)
if len(idx) >= K:
    top = idx[np.argsort(mag[idx])[-K:]]
else:
    top = np.linspace(1, T // 2 - 1, K).astype(int)
omega = np.sort(freqs[T // 2:][top])

# --- ADMM-Iteration ---
u_hat = np.zeros((K, T), dtype=complex)
lamb = np.zeros(T, dtype=complex)

for it in range(n_iter):
    u_prev = u_hat.copy()
    sum_u = u_hat.sum(axis=0)
    for k in range(K):
        rest = f_hat_plus - sum_u + u_hat[k]
        u_hat[k] = (rest - lamb / 2) / (1 + 2 * alpha * (freqs - omega[k]) ** 2)
        sum_u = sum_u - u_prev[k] + u_hat[k]
        pos = slice(T // 2, T)
        leistung = np.abs(u_hat[k, pos]) ** 2
        omega[k] = np.sum(freqs[pos] * leistung) / (np.sum(leistung) + 1e-12)
    lamb = lamb + tau * (u_hat.sum(axis=0) - f_hat_plus)
    if np.sum(np.abs(u_hat - u_prev) ** 2) / T < tol:
        break

# --- Moden in den Zeitbereich zurueck (hermitesch spiegeln) ---
u_full = np.zeros((K, T), dtype=complex)
u_full[:, T // 2:] = u_hat[:, T // 2:]
u_full[:, 1:T // 2] = np.conj(u_hat[:, -1:T // 2:-1])
moden = np.real(np.fft.ifft(np.fft.ifftshift(u_full, axes=1), axis=1))
moden = moden[:, N // 2:N // 2 + N]             # Spiegelung entfernen

# --- Zuordnung: Moden nahe einer Harmonischen von f_r -> Nutzschall ---
omega_hz = np.abs(omega) * sr
harmonisch = np.round(omega_hz / f_r) * f_r
ist_nutz = np.abs(omega_hz - harmonisch) <= (f_r / 2)

ns = moden[ist_nutz].sum(axis=0)
hs = moden[~ist_nutz].sum(axis=0)

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Mittenfrequenzen (Hz): {np.round(np.sort(omega_hz), 1)}")
print(f"Nutzschall-Moden: {np.where(ist_nutz)[0].tolist()}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")