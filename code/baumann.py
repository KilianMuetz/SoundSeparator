
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, find_peaks

# --- Parameter (Werte fuer Musik nach Baumann Tab. S. 94) ---
BASE = Path(__file__).parent
eingabe = BASE / "../data/observ_1.wav"
ausgabe_ns = BASE / "../sep/konturierung/KonturNs.wav"   # tonal
ausgabe_hs = BASE / "../sep/konturierung/KonturHs.wav"   # geraeuschhaft

delta_f_cent = 80.0     # max. Frequenzunterschied benachbarter TT (Cent)
delta_l_db = 6.0        # max. Pegelunterschied benachbarter TT (dB)
t_min_ms = 80.0         # min. Liniendauer -> tonal (ms)
t_lueck_ms = 40.0       # max. Dauer einer geschlossenen Linienunterbrechung (ms)
peak_min_db = -60.0     # Peak-Schwelle relativ zum Maximum (dB)

nperseg = 1024
noverlap = 512

# --- Laden & STFT ---
y, sr = sf.read(eingabe, dtype="float64")
f, t, Y = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
Y_mag, Y_phase = np.abs(Y), np.angle(Y)

n_freq, n_frames = Y_mag.shape
ta_ms = (t[1] - t[0]) * 1000.0             # Auswerteintervall TA in ms
lueck_frames = int(round(t_lueck_ms / ta_ms))
min_frames = int(round(t_min_ms / ta_ms))

# --- Pegel in dB, Peak-Schwelle relativ zum Gesamtmaximum ---
Y_db = 20 * np.log10(Y_mag + 1e-12)
schwelle = Y_db.max() + peak_min_db

# --- 1) Peak-Picking je Frame -> Teiltoene (Frequenz-Bin-Indizes) ---
peaks_pro_frame = []
for k in range(n_frames):
    idx, _ = find_peaks(Y_db[:, k], height=schwelle)
    peaks_pro_frame.append(list(idx))

# --- Hilfsfunktion: Frequenzabstand in Cent ---
def cent(f1, f2):
    return abs(1200.0 * np.log2((f2 + 1e-12) / (f1 + 1e-12)))

# --- 2) Teiltoene ueber Zeit zu Linien verketten ---
# tonale Maske: markiert (freq_bin, frame), die zu einer gueltigen Linie gehoeren
tonal_mask = np.zeros_like(Y_mag, dtype=bool)
belegt = [set() for _ in range(n_frames)]   # schon verkettete Peaks

for k0 in range(n_frames):
    for p0 in peaks_pro_frame[k0]:
        if p0 in belegt[k0]:
            continue
        # neue Linie starten
        linie = [(k0, p0)]
        k, p = k0, p0
        luecke = 0
        while k + 1 < n_frames:
            k_next = k + 1
            # besten Nachfolger im naechsten Frame suchen (Freq+Pegel-Naehe)
            bester, best_d = None, None
            for pn in peaks_pro_frame[k_next]:
                if pn in belegt[k_next]:
                    continue
                if cent(f[p], f[pn]) <= delta_f_cent \
                        and abs(Y_db[p, k] - Y_db[pn, k_next]) <= delta_l_db:
                    d = cent(f[p], f[pn])
                    if best_d is None or d < best_d:
                        bester, best_d = pn, d
            if bester is not None:
                linie.append((k_next, bester))
                k, p, luecke = k_next, bester, 0
            else:
                luecke += 1
                if luecke > lueck_frames:
                    break
                k = k_next   # Luecke ueberbruecken (Kontinuitaetseffekt)

        # Linie nur behalten, wenn lang genug -> tonal
        if len(linie) >= min_frames:
            for (kk, pp) in linie:
                tonal_mask[pp, kk] = True
                belegt[kk].add(pp)

# --- 3) Maske auf Frequenz-Nachbarschaft verbreitern (Bin-Energie zuordnen) ---
# jeder tonale Peak beansprucht sein Bin +/-1, damit die Rekonstruktion Energie behaelt
mask = tonal_mask.copy()
mask[1:, :]  |= tonal_mask[:-1, :]
mask[:-1, :] |= tonal_mask[1:, :]

# --- Trennung ---
Ns = Y * mask
Hs = Y * (~mask)

# --- Rekonstruktion ---
_, ns = istft(Ns, fs=sr, nperseg=nperseg, noverlap=noverlap)
_, hs = istft(Hs, fs=sr, nperseg=nperseg, noverlap=noverlap)
ns, hs = ns[:len(y)], hs[:len(y)]

# --- Speichern ---
ausgabe_ns.parent.mkdir(parents=True, exist_ok=True)
sf.write(ausgabe_ns, ns, sr)
sf.write(ausgabe_hs, hs, sr)
print(f"TA = {ta_ms:.1f} ms, min. Liniendauer = {min_frames} Frames")
print(f"Tonale Bins: {tonal_mask.sum()} von {Y_mag.size}")
print(f"Fertig: {ausgabe_ns}")
print(f"Fertig: {ausgabe_hs}")