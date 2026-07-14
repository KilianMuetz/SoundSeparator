from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/cepstrum/CepNs.wav"   # periodisch
ausgabe_hs = BASE / "../sep/cepstrum/CepHs.wav"   # aperiodisch

f_r = 48.0        # Grundfrequenz der Rotation in Hz (bestimmt die Grundquefrenz)
breite = 3        # halbe Lifter-Breite je Rahmonic-Peak (Quefrenz-Bins)
nperseg = 1024
noverlap = 512

# --- Laden & STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)
n_freq = Y_mag.shape[0]

# --- Reelles Cepstrum je Frame (entlang Frequenzachse) ---
logmag = np.log(Y_mag + 1e-10)
cep = np.fft.irfft(logmag, axis=0)
n_cep = cep.shape[0]

# --- Grundquefrenz und Liftering der Rahmonic-Peaks ---
q0 = n_cep / ((sr / 2) / f_r)          # Quefrenz-Index der Grundperiode
cep_edit = cep.copy()
if q0 >= 2:
    m = 1
    while int(round(m * q0)) < n_cep // 2:
        qi = int(round(m * q0))
        cep_edit[max(0, qi - breite):qi + breite + 1, :] = 0          # Rahmonic
        cep_edit[n_cep - qi - breite:n_cep - qi + breite + 1, :] = 0  # sym. Teil
        m += 1

# --- Rueckwandlung -> aperiodische Magnitude ---
logmag_ap = np.real(np.fft.rfft(cep_edit, axis=0))[:n_freq]
Y_mag_ap = np.clip(np.exp(logmag_ap), 0, Y_mag)   # nicht groesser als Original

# --- Trennung ---
Ns_mag = Y_mag - Y_mag_ap                          # periodischer Anteil

_, ns = istft(Ns_mag * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Y_mag_ap * np.exp(1j * Y_phase), fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"f_r = {f_r} Hz, Grundquefrenz q0 = {q0:.1f}, Lifter-Breite = {breite}")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")