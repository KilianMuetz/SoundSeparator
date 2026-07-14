"""
Matching Pursuit (Mallat & Zhang, 1993)
Zerlegt das Signal greedy in Atome aus einem ueberkompletten Dictionary: in
jedem Schritt wird das Atom mit der hoechsten Korrelation zum aktuellen
Residuum gewaehlt, sein Beitrag abgezogen und der Vorgang wiederholt (sparse
Approximation). Als Dictionary dienen Fourier-/Gabor-Atome (Sinuskomponenten);
die Korrelation mit allen Atomen wird effizient per FFT berechnet.

Tonale, drehzahlsynchrone Maschinen-Anteile werden von wenigen starken Atomen
erfasst, breitbandiges Rauschen und Transienten nicht. Zuordnung ueber die
Atom-Frequenz (konsistent mit VMD/NMF/EMD): Atome im Maschinenband
(<= f_grenze) = Nutzschall, der Rest verbleibt im Stoerschall.

Die Verarbeitung erfolgt blockweise mit Overlap-Add (Hann-Fenster, 50 %
Ueberlappung): MP wird je Block gerechnet, die Blockergebnisse werden
gefenstert ueberlappend zusammengesetzt und durch die Fenstersumme normiert.
Dies vermeidet die hoerbaren Diskontinuitaeten ("Ticken") an harten
Blockgrenzen.
  Ns = Summe der Maschinenband-Atome            = Nutzschall
  Hs = Residuum y - Ns                          = Stoerschall
"""

from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parameter ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/matching_pursuit/MpNs.wav"
ausgabe_hs = BASE / "../sep/matching_pursuit/MpHs.wav"

f_grenze = 800.0    # Obergrenze des Maschinenbands in Hz (Atom-Zuordnung)
n_atoms = 100       # Anzahl greedy gewaehlter Atome pro Block
seg_len = 4096      # Blocklaenge
hop = 2048          # Blockversatz (seg_len/2 = 50 % Overlap)

# --- Laden ---
y, sr = sf.read(eingabe, dtype="float64")
N = len(y)

# --- Matching Pursuit je Block + Overlap-Add ---
fenster = np.hanning(seg_len)
ns = np.zeros(N)
norm = np.zeros(N)                       # Fenster-Ueberlappungssumme zur Normierung

for start in range(0, N - seg_len + 1, hop):
    seg = y[start:start + seg_len]       # MP auf dem rohen Block
    L = len(seg)
    freqs = np.fft.rfftfreq(L, 1 / sr)
    r = seg.copy()
    ns_block = np.zeros(L)
    for _ in range(n_atoms):
        R = np.fft.rfft(r)
        idx = np.argmax(np.abs(R))       # staerkstes Atom = groesster Frequenzbin
        atom_spec = np.zeros_like(R)
        atom_spec[idx] = R[idx]
        atom = np.fft.irfft(atom_spec, n=L)
        if freqs[idx] <= f_grenze:        # Zuordnung nach Atom-Frequenz
            ns_block += atom
        r -= atom
    # gefenstertes Overlap-Add
    ns[start:start + seg_len] += ns_block * fenster
    norm[start:start + seg_len] += fenster

norm[norm < 1e-8] = 1.0
ns = ns / norm

# --- Residuum = Stoerschall ---
hs = y - ns

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"n_atoms = {n_atoms}/Block, Overlap-Add (hop={hop}), f_grenze = {f_grenze} Hz")
print(f"Nutz-Energie / Gesamt: {np.sum(ns**2)/np.sum(y**2):.3f}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")