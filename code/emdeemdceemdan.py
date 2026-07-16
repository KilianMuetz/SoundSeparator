from pathlib import Path

import numpy as np
import soundfile as sf
from PyEMD import EMD, EEMD, CEEMDAN

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"

verfahren = "CEEMDAN"   # "EMD", "EEMD" oder "CEEMDAN"
f_grenze = 800.0        # Obergrenze des Maschinenbands in Hz (Zuordnung)
max_imf = 10            # max. Anzahl IMFs
ensemble_trials = 50    # EEMD/CEEMDAN: Anzahl Rausch-Durchlaeufe

# --- Ausgabepfade je Verfahren (kein gegenseitiges Ueberschreiben) ---
ordner = {"EMD": "emd", "EEMD": "eemd", "CEEMDAN": "ceemdan"}[verfahren]
praefix = {"EMD": "emd", "EEMD": "eemd", "CEEMDAN": "ceemdan"}[verfahren]
ausgabe_ns = BASE / f"../sep/{ordner}/{praefix}Ns.wav"
ausgabe_hs = BASE / f"../sep/{ordner}/{praefix}Hs.wav"

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Zerlegung in IMFs ---
if verfahren == "EEMD":
    zerleger = EEMD(trials=ensemble_trials, parallel=False)
    zerleger.noise_seed(42)
    imfs = zerleger.eemd(y, max_imf=max_imf)
elif verfahren == "CEEMDAN":
    zerleger = CEEMDAN(trials=ensemble_trials, parallel=False)
    zerleger.noise_seed(42)
    imfs = zerleger.ceemdan(y, max_imf=max_imf)
else:
    zerleger = EMD()
    imfs = zerleger.emd(y, max_imf=max_imf)

# --- Zuordnung ueber den spektralen Schwerpunkt jeder IMF ---
freqs = np.fft.rfftfreq(N, 1 / sr)
ns = np.zeros(N)
hs = np.zeros(N)
for imf in imfs:
    spec = np.abs(np.fft.rfft(imf))
    centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-12)
    if centroid <= f_grenze:
        ns += imf
    else:
        hs += imf

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"{verfahren}: {imfs.shape[0]} IMFs, f_grenze = {f_grenze} Hz")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")