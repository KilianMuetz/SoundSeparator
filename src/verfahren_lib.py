"""
verfahren_lib.py — Die ausgewaehlten Trennverfahren als aufrufbare Funktionen.

Jede Funktion hat die Signatur  trenne(y, sr) -> (ns, hs)
  ns = geschaetzter Nutzschall  (Maschine)
  hs = geschaetzter Stoerschall (Rest)
Es gilt stets  len(ns) == len(hs) == len(y).

Das Referenzspektrum wird aus dem Mischsignal selbst geschaetzt, als
20. Perzentil je Frequenz ueber alle Zeitfenster (Stahl et al. 2000).
"""

import librosa
import numpy as np
from PyEMD import EMD
from scipy.signal import istft, stft
from scipy.special import i0e, i1e
from sklearn.decomposition import NMF

# --- Gemeinsame STFT-Parameter ---
NPERSEG = 1024
NOVERLAP = 512
EPS = 1e-12

# --- Gemeinsame Verfahrensparameter ---
PERZENTIL = 20        # Perzentil je Frequenzband fuer die blinde Grundspektrumschaetzung
SIM_SCHWELLE = 0.6    # Mindestaehnlichkeit einer Komponente zum Grundspektrum


def _stft(y, sr):
    """Spektrogramm: Betrag und Phase."""
    _, _, Y = stft(y, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return np.abs(Y), np.angle(Y)


def _istft(mag, phase, sr, n):
    _, x = istft(mag * np.exp(1j * phase), fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    x = x[:n]
    if len(x) < n:
        x = np.concatenate([x, np.zeros(n - len(x))])
    return x


def _grundspektrum(Y_mag):
    """Blinde Schaetzung des stationaeren Grundspektrums aus dem Mischsignal.

    Perzentil je Frequenzband ueber alle Frames: kurzzeitige Ereignisse
    heben das Perzentil kaum an, dauerhaft vorhandene Anteile bestimmen es.
    """
    return np.percentile(Y_mag, PERZENTIL, axis=1, keepdims=True)


def _mittelspektrum(x, sr):
    """Mittleres Betragsspektrum eines Signals auf demselben Frequenzraster."""
    return _stft(x, sr)[0].mean(axis=1)


def _aehnlichkeit(spektren, referenz):
    """Kosinus-Aehnlichkeit von Komponentenspektren (spaltenweise) zum Referenzspektrum."""
    r = referenz / (np.linalg.norm(referenz) + EPS)
    s = spektren / (np.linalg.norm(spektren, axis=0, keepdims=True) + EPS)
    return s.T @ r


def _zurueck(Ns_mag, Hs_mag, phase, sr, n):
    """Beide Teilspektrogramme mit der Phase des Mischsignals zurueck in Zeitsignale."""
    return _istft(Ns_mag, phase, sr, n), _istft(Hs_mag, phase, sr, n)


def _ist_nutz(sim):
    """Komponenten mit ausreichender Aehnlichkeit zum Referenzspektrum sind Nutzschall."""
    maske = sim >= SIM_SCHWELLE
    if not maske.any():
        maske[np.argmax(sim)] = True   # mindestens eine Komponente
    return maske


# ---------------------------------------------------------------- physikalisch
def synchrone_mittelung(y, sr, block_s=0.5, rpm_min=1500, rpm_max=5000):
    """Synchrone Mittelung nach Bechhoefer und Kingsley (2009).

    Das Muster, das sich mit jeder Umdrehung wiederholt, ist der Nutzschall.
    Gearbeitet wird in Bloecken, da die Drehzahl schwanken kann.
    """
    lag_min = int(sr * 60 / rpm_max)   # kuerzeste Umdrehung in Abtastwerten
    lag_max = int(sr * 60 / rpm_min)   # laengste Umdrehung in Abtastwerten
    block = int(block_s * sr)

    ns = y.copy()                      # Rest ohne volle Umdrehung bleibt unveraendert
    for start in range(0, len(y), block):
        blk = y[start:start + block]
        if len(blk) < 2 * lag_max:     # zu kurz fuer zwei Umdrehungen
            continue

        # Periode = Maximum der Autokorrelation im Drehzahlbereich
        akf = np.correlate(blk, blk, mode="full")[len(blk) - 1:]
        periode = lag_min + np.argmax(akf[lag_min:lag_max])

        # Umdrehungen untereinanderlegen und spaltenweise mitteln
        anzahl = len(blk) // periode
        mittel = blk[:anzahl * periode].reshape(anzahl, periode).mean(axis=0)
        ns[start:start + anzahl * periode] = np.tile(mittel, anzahl)

    return ns, y - ns


# ------------------------------------------------ statistische Spektralschaetzung
def spektralsubtraktion(y, sr, beta=0.04):
    """Spektralsubtraktion (Boll 1979). Untergrenze beta nach Stahl et al. (2000)."""
    Y_mag, phase = _stft(y, sr)
    Hs_mag = np.maximum(Y_mag - _grundspektrum(Y_mag), beta * Y_mag)
    return _zurueck(Y_mag - Hs_mag, Hs_mag, phase, sr, len(y))


def _gain_schleife(Y_mag, gain, alpha_dd):
    """Gemeinsame Schleife von Wiener und MMSE-STSA (Decision-Directed, Ephraim und Malah 1984).

    Die Referenz ist die Maschine, die Verstaerkung extrahiert daher den Stoerschall.
    """
    ref_pow = _grundspektrum(Y_mag) ** 2 + EPS
    Y_pow = Y_mag ** 2
    Hs_mag = np.zeros_like(Y_mag)
    vorher = Y_mag[:, [0]]
    for k in range(Y_mag.shape[1]):
        gamma = Y_pow[:, [k]] / ref_pow
        xi = alpha_dd * (vorher ** 2 / ref_pow) + (1 - alpha_dd) * np.maximum(gamma - 1, 0)
        Hs_mag[:, [k]] = gain(xi, gamma) * Y_mag[:, [k]]
        vorher = Hs_mag[:, [k]]
    return Hs_mag


def wiener(y, sr, alpha_dd=0.98):
    """Wiener-Filter (Abd El-Fattah et al. 2014)."""
    Y_mag, phase = _stft(y, sr)
    gain = lambda xi, gamma: xi / (1 + xi)
    Hs_mag = _gain_schleife(Y_mag, gain, alpha_dd)
    return _zurueck(Y_mag - Hs_mag, Hs_mag, phase, sr, len(y))


def mmse_stsa(y, sr, alpha_dd=0.98):
    """MMSE-STSA (Ephraim und Malah 1984)."""
    Y_mag, phase = _stft(y, sr)

    def gain(xi, gamma):
        v = xi / (1 + xi) * gamma
        g = (np.sqrt(np.pi) / 2) * (np.sqrt(v) / np.maximum(gamma, EPS)) \
            * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
        return np.minimum(g, 1.0)

    Hs_mag = _gain_schleife(Y_mag, gain, alpha_dd)
    return _zurueck(Y_mag - Hs_mag, Hs_mag, phase, sr, len(y))


# ------------------------------------------------------- Struktur/Faktorisierung
def hpss(y, sr, filterlaenge=17, p=2.0):
    """HPSS (FitzGerald 2010, S. 3) mit librosa. Harmonisch = Maschine, perkussiv = Stoerschall."""
    Y_mag, phase = _stft(y, sr)
    maske_h, maske_p = librosa.decompose.hpss(Y_mag, kernel_size=filterlaenge, power=p, mask=True)
    return _zurueck(Y_mag * maske_h, Y_mag * maske_p, phase, sr, len(y))


def nmf(y, sr, K=8, seed=0):
    """NMF (Virtanen 2007) mit scikit-learn, KL-Divergenz.

    Zuordnung der Komponenten ueber die Aehnlichkeit zum Referenzspektrum.
    """
    Y_mag, phase = _stft(y, sr)
    modell = NMF(n_components=K, beta_loss="kullback-leibler", solver="mu",
                 max_iter=1000, init="random", random_state=seed)   # Ende ueber Standardtoleranz
    W = modell.fit_transform(Y_mag)   # Spektren der Komponenten
    H = modell.components_            # Verlauf der Komponenten ueber die Zeit

    nutz = _ist_nutz(_aehnlichkeit(W, _grundspektrum(Y_mag).ravel()))
    WH = W @ H + EPS
    Ns_mag = (W[:, nutz] @ H[nutz]) / WH * Y_mag
    Hs_mag = (W[:, ~nutz] @ H[~nutz]) / WH * Y_mag
    return _zurueck(Ns_mag, Hs_mag, phase, sr, len(y))


def rpca(y, sr, tol=1e-7, max_iter=200):
    """RPCA ueber Inexact ALM (Candes et al. 2011). Niedrigrang = Maschine, duenn besetzt = Stoerschall."""
    M, phase = _stft(y, sr)
    lam = 1.0 / np.sqrt(max(M.shape))
    norm2 = np.linalg.norm(M, 2)
    mu, mu_max, rho = 1.25 / norm2, 1.25e7 / norm2, 1.5

    S = np.zeros_like(M)
    Yd = M / max(norm2, np.linalg.norm(M, np.inf) / lam)
    for _ in range(max_iter):
        U, sig, Vt = np.linalg.svd(M - S + Yd / mu, full_matrices=False)
        L = (U * np.maximum(sig - 1 / mu, 0)) @ Vt                   # Niedrigrang-Anteil
        T = M - L + Yd / mu
        S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0)          # duenn besetzter Anteil
        Z = M - L - S
        Yd += mu * Z
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(Z) / np.linalg.norm(M) < tol:
            break
    return _zurueck(L, S, phase, sr, len(y))


# ----------------------------------------------------------- adaptive Zerlegung
def emd(y, sr):
    """EMD (Huang et al. 1998) mit PyEMD, ohne Begrenzung der Modenzahl.

    Zuordnung der Moden ueber die Aehnlichkeit zum Referenzspektrum.
    """
    imfs = EMD().emd(y)

    spektren = np.column_stack([_mittelspektrum(imf, sr) for imf in imfs])
    nutz = _ist_nutz(_aehnlichkeit(spektren, _grundspektrum(_stft(y, sr)[0]).ravel()))

    hs = imfs[~nutz].sum(axis=0)
    return y - hs, hs   # Nutzschall = alles ausser Stoerschall, inkl. Residuum


# ------------------------------------------- nur Prototypen (Trennschaerfe)
def ssa(y, sr, L=300, r=5, block=8000):
    """SSA (Hassani 2007). Die r staerksten Komponenten sind der Nutzschall."""
    ns = y.copy()                                   # zu kurzer Restblock bleibt unveraendert
    for start in range(0, len(y), block):
        seg = y[start:start + block]
        if len(seg) < L + 10:
            continue
        K = len(seg) - L + 1
        X = np.column_stack([seg[i:i + L] for i in range(K)])   # verschobene Ausschnitte
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        X_r = (U[:, :r] * S[:r]) @ Vt[:r]                         # staerkste r Komponenten
        summe, anzahl = np.zeros(len(seg)), np.zeros(len(seg))   # zurueck in ein Zeitsignal
        for i in range(L):
            summe[i:i + K] += X_r[i]
            anzahl[i:i + K] += 1
        ns[start:start + len(seg)] = summe / anzahl
    return ns, y - ns


def matching_pursuit(y, sr, bausteine=100, f_grenze=800.0, block=4096):
    """Matching Pursuit (Mallat und Zhang 1993). Bausteine unter f_grenze sind der Nutzschall."""
    versatz = block // 2
    fenster = np.hanning(block)
    freqs = np.fft.rfftfreq(block, 1 / sr)
    ns, gewicht = np.zeros_like(y), np.zeros_like(y)
    for start in range(0, len(y) - block + 1, versatz):
        rest = y[start:start + block].copy()
        ns_block = np.zeros(block)
        for _ in range(bausteine):
            R = np.fft.rfft(rest)
            idx = np.argmax(np.abs(R))                 # staerkster Baustein
            spek = np.zeros_like(R)
            spek[idx] = R[idx]
            baustein = np.fft.irfft(spek, n=block)
            if freqs[idx] <= f_grenze:
                ns_block += baustein
            rest -= baustein
        ns[start:start + block] += ns_block * fenster
        gewicht[start:start + block] += fenster
    ns = ns / np.where(gewicht < 1e-8, 1.0, gewicht)
    return ns, y - ns



# --- Registry: Name -> Funktion (Reihenfolge wie in Tabelle 4.2) ---
VERFAHREN = {
    "synchronous_averaging": synchrone_mittelung,
    "wiener": wiener,
    "spectralsubtraction": spektralsubtraktion,
    "mmse_stsa": mmse_stsa,
    "nmf": nmf,
    "hpss": hpss,
    "rpca": rpca,
    "emd": emd,
}

# --- Prototypen fuer die Trennschaerfe: die acht Verfahren plus SSA und Matching Pursuit ---
PROTOTYPEN = {**VERFAHREN, "ssa": ssa, "matching_pursuit": matching_pursuit}
