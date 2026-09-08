from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/demucs/demucsNs.wav"   # gewaehlter Stem
ausgabe_hs = BASE / "../sep/demucs/demucsHs.wav"   # Rest

modellname = "htdemucs"     # "htdemucs", "htdemucs_ft" (besser, langsamer), "htdemucs_6s"
nutz_stem = "other"         # welcher Stem als Nutzschall gilt: vocals/drums/bass/other
geraet = "cpu"              # "cpu" oder "cuda"

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float32")
if y.ndim == 1:
    y = np.stack([y, y])        # Demucs erwartet Stereo (2 Kanaele)
else:
    y = y.T
wav = torch.from_numpy(y)

# --- Modell laden und anwenden ---
modell = get_model(modellname)
modell.to(geraet)
# Normierung wie in Demucs ueblich
ref = wav.mean(0)
wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)
quellen = apply_model(modell, wav_norm[None], device=geraet, progress=True)[0]
quellen = quellen * ref.std() + ref.mean()

# --- Stems nach Namen zuordnen ---
stem_namen = modell.sources          # z.B. ['drums','bass','other','vocals']
idx = stem_namen.index(nutz_stem)
ns_stereo = quellen[idx].cpu().numpy()
hs_stereo = np.zeros_like(ns_stereo)
for i in range(len(stem_namen)):
    if i != idx:
        hs_stereo += quellen[i].cpu().numpy()

# --- auf Mono zurueck (Mittel beider Kanaele) ---
ns = ns_stereo.mean(0)
hs = hs_stereo.mean(0)

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"Modell: {modellname}, Stems: {stem_namen}")
print(f"Nutzschall-Stem: '{nutz_stem}'")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")