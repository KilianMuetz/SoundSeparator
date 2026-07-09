from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


# =====================================
# Einstellungen
# =====================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_FILE = BASE_DIR.parent / "data" / "observ_1.wav"

OUTPUT_DIR = PROJECT_DIR / "sep" / "spectralsubtraction"

# Ausgabedateien
OUTPUT_NS = OUTPUT_DIR / "SpecSubNs.wav"
OUTPUT_HS = OUTPUT_DIR / "SpecSubHs.wav"
OUTPUT_SS = OUTPUT_DIR / "SpecSubSs.wav"

N_FFT = 2048
HOP = 512

NOISE_DURATION = 0.5      # Sekunden
ALPHA = 2.2               # Oversubtraction
BETA = 0.02               # Spectral Floor

IMPACT_WINDOW = 0.08      # ±80 ms


# =====================================
# Datei laden
# =====================================

y, sr = sf.read(INPUT_FILE)

if y.ndim > 1:
    y = np.mean(y, axis=1)


# =====================================
# STFT
# =====================================

D = librosa.stft(
    y,
    n_fft=N_FFT,
    hop_length=HOP
)

magnitude = np.abs(D)
phase = np.exp(1j * np.angle(D))


# =====================================
# Noise-Profil
# =====================================

noise_frames = max(
    1,
    int(NOISE_DURATION * sr / HOP)
)

noise_profile = np.mean(
    magnitude[:, :noise_frames],
    axis=1,
    keepdims=True
)


# =====================================
# Klassische Spektralsubtraktion
# =====================================

clean_mag = np.maximum(
    magnitude - ALPHA * noise_profile,
    BETA * magnitude
)


# =====================================
# Rekonstruktion
# =====================================

foreground = librosa.istft(
    clean_mag * phase,
    hop_length=HOP,
    length=len(y)
)

background = y - foreground


# =====================================
# Schlag extrahieren
# =====================================

onset_env = librosa.onset.onset_strength(
    y=y,
    sr=sr
)

times = librosa.times_like(
    onset_env,
    sr=sr
)

impact_time = times[np.argmax(onset_env)]

impact = np.zeros_like(y)

start = max(
    0,
    int((impact_time - IMPACT_WINDOW) * sr)
)

end = min(
    len(y),
    int((impact_time + IMPACT_WINDOW) * sr)
)

impact[start:end] = y[start:end]


# =====================================
# Hintergrund um Schlag bereinigen
# =====================================

background[start:end] = 0


# =====================================
# Normalisierung
# =====================================

def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / (peak * 1.01)
    return signal


foreground = normalize(foreground)
background = normalize(background)
impact = normalize(impact)


# =====================================
# Speichern
# =====================================

sf.write(OUTPUT_NS, foreground, sr)
sf.write(OUTPUT_HS, background, sr)
sf.write(OUTPUT_SS, impact, sr)

print("Fertig.")
print("Erzeugte Dateien:")
print(" -", OUTPUT_NS)
print(" -", OUTPUT_HS)
print(" -", OUTPUT_SS)