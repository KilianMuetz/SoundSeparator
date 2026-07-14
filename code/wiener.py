"""
Wiener-Filter (spektral, MMSE-optimal)
Wendet pro Frequenzbin eine weiche, SNR-abhaengige Verstaerkung an:
  G = xi / (1 + xi)        mit xi = a-priori-SNR
Wo Nutzsignal dominiert (hohes SNR) laesst der Filter durch, wo Rauschen
dominiert daempft er. Das a-priori-SNR wird ueber den Decision-Directed-Ansatz
(Ephraim & Malah) aus der vorigen Schaetzung gebildet -> weniger Musical Noise
als bei harter Spektralsubtraktion.

Der Wiener-Gain ist einfacher als die MMSE-STSA-Verstaerkung (kein Bessel),
liefert aber dieselbe weiche, komplementaere Zerlegung:
  Ns = G * Y            (Anteil ueber dem Rauschpegel)   = Nutzschall
  Hs = (1 - G) * Y      (unterdrueckter Anteil)          = Stoerschall
Annahme: Am Anfang liegt ein reines Rauschsegment (Referenz) vor.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/wiener/WienerNs.wav"   # Nutzschall
ausgabe_hs = BASE / "../sep/wiener/WienerHs.wav"   # Hintergrundschall

rauschreferenz_ende_s = 1.0   # erste Sekunde = Rauschreferenz
alpha_dd = 0.98               # Decision-Directed-Glaettung
xi_min = 10 ** (-25 / 10)     # a-priori-SNR-Floor (-25 dB)
nperseg = 1024
noverlap = 512

# --- Laden & STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)
Y_pow = Y_mag ** 2

# --- Rauschleistung aus Referenzsegment schaetzen ---
ref_frames = t <= rauschreferenz_ende_s
noise_pow = np.mean(Y_pow[:, ref_frames], axis=1, keepdims=True)

# --- Frame-weise Wiener-Verstaerkung mit Decision-Directed-SNR ---
Ns_mag = np.zeros_like(Y_mag)
prev_amp = Y_mag[:, [0]]

for k in range(Y_mag.shape[1]):
    gamma = Y_pow[:, [k]] / noise_pow                  # a-posteriori-SNR
    xi = alpha_dd * (prev_amp ** 2 / noise_pow) \
         + (1 - alpha_dd) * np.maximum(gamma - 1, 0)   # a-priori-SNR (DD)
    xi = np.maximum(xi, xi_min)
    gain = xi / (1 + xi)                               # Wiener-Gain
    Ns_mag[:, [k]] = gain * Y_mag[:, [k]]
    prev_amp = Ns_mag[:, [k]]

# --- Hintergrundschall: komplementaerer Rest ---
Hs_mag = Y_mag - Ns_mag

# --- Rekonstruktion mit Originalphase ---
_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")