"""
verfahren_lib.py — Die ausgewaehlten Trennverfahren als aufrufbare Funktionen.

Jede Funktion hat die Signatur  trenne(y, sr) -> (ns, hs)
  ns = geschaetzter Nutzschall  (Maschine)
  hs = geschaetzter Stoerschall (Rest)
Es gilt stets  len(ns) == len(hs) == len(y).

Die zuordnenden Verfahren schaetzen das stationaere Grundspektrum blind
aus dem Mischsignal selbst (Perzentil je Frequenzband ueber alle Frames,
vereinfachte Minimum-Statistics-Schaetzung). Sie erhalten damit keine
zusaetzliche Information ueber die Quelle.
"""

import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import median_filter
from scipy.special import i0e, i1e

# --- Gemeinsame STFT-Parameter ---
NPERSEG = 1024
NOVERLAP = 512
EPS = 1e-12

# --- Gemeinsame Verfahrensparameter ---
PERZENTIL = 20        # Perzentil je Frequenzband fuer die blinde Grundspektrumschaetzung
SIM_SCHWELLE = 0.6    # Mindestaehnlichkeit einer Komponente zum Grundspektrum


def _stft(y, sr):
    f, t, Y = stft(y, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return f, t, np.abs(Y), np.angle(Y)


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
    _, _, mag, _ = _stft(x, sr)
    return mag.mean(axis=1)


def _aehnlichkeit(spektren, referenz):
    """Kosinus-Aehnlichkeit von Komponentenspektren (spaltenweise) zum Referenzspektrum."""
    r = referenz / (np.linalg.norm(referenz) + EPS)
    s = spektren / (np.linalg.norm(spektren, axis=0, keepdims=True) + EPS)
    return s.T @ r


# ---------------------------------------------------------------- physikalisch
def synchrone_mittelung(y, sr, block_s=0.5, rpm_min=1500, rpm_max=5000):
    """Synchrone Mittelung (McFadden 1987): periodischer Anteil = Nutzschall.

    Blockweise, da die Drehzahl des Ventilators ueber 10 s driftet und eine
    Mittelung ueber die volle Laenge den periodischen Anteil ausloescht.
    """
    lag_min = int(sr / (rpm_max / 60.0))
    lag_max = int(sr / (rpm_min / 60.0))
    B = int(block_s * sr)

    ns = np.zeros(len(y))
    for start in range(0, len(y), B):
        blk = y[start:start + B]
        if len(blk) < 2 * lag_max:
            ns[start:start + len(blk)] = blk
            continue

        akf = np.correlate(blk, blk, mode="full")[len(blk) - 1:]
        i = lag_min + int(np.argmax(akf[lag_min:lag_max]))
        a, b, c = akf[i - 1], akf[i], akf[i + 1]
        nenner = a - 2 * b + c
        delta = 0.5 * (a - c) / nenner if abs(nenner) > EPS else 0.0
        periode = max(int(round(i + delta)), 2)

        anzahl = len(blk) // periode
        mittel = blk[:anzahl * periode].reshape(anzahl, periode).mean(axis=0)
        rek = np.concatenate([np.tile(mittel, anzahl), blk[anzahl * periode:]])
        ns[start:start + len(blk)] = rek[:len(blk)]

    return ns, y - ns


# ------------------------------------------------ statistische Spektralschaetzung
def wiener(y, sr, alpha_dd=0.98, xi_min=10 ** (-25 / 10)):
    """Wiener-Filter. Referenz = Maschine, Verstaerkung extrahiert den Stoerschall."""
    f, t, Y_mag, Y_phase = _stft(y, sr)
    Y_pow = Y_mag ** 2

    grund_pow = _grundspektrum(Y_mag) ** 2 + EPS

    Hs_mag = np.zeros_like(Y_mag)
    prev_amp = Y_mag[:, [0]]
    for k in range(Y_mag.shape[1]):
        gamma = Y_pow[:, [k]] / grund_pow
        xi = alpha_dd * (prev_amp ** 2 / grund_pow) \
             + (1 - alpha_dd) * np.maximum(gamma - 1, 0)
        xi = np.maximum(xi, xi_min)
        gain = xi / (1 + xi)
        Hs_mag[:, [k]] = gain * Y_mag[:, [k]]
        prev_amp = Hs_mag[:, [k]]

    Ns_mag = Y_mag - Hs_mag
    return _istft(Ns_mag, Y_phase, sr, len(y)), _istft(Hs_mag, Y_phase, sr, len(y))


def spektralsubtraktion(y, sr, alpha=1.0, beta=0.02):
    """Spektralsubtraktion (Boll 1979). Harte Subtraktion mit Spectral Floor."""
    f, t, Y_mag, Y_phase = _stft(y, sr)

    grund_mag = _grundspektrum(Y_mag)

    Hs_mag = np.maximum(Y_mag - alpha * grund_mag, beta * Y_mag)
    Ns_mag = Y_mag - Hs_mag
    return _istft(Ns_mag, Y_phase, sr, len(y)), _istft(Hs_mag, Y_phase, sr, len(y))


def mmse_stsa(y, sr, alpha_dd=0.98, xi_min=10 ** (-25 / 10)):
    """MMSE-STSA (Ephraim & Malah 1984). Referenz = Maschine wie beim Wiener-Filter."""
    f, t, Y_mag, Y_phase = _stft(y, sr)
    Y_pow = Y_mag ** 2

    grund_pow = _grundspektrum(Y_mag) ** 2 + EPS

    Hs_mag = np.zeros_like(Y_mag)
    prev_amp = Y_mag[:, [0]]
    for k in range(Y_mag.shape[1]):
        gamma = Y_pow[:, [k]] / grund_pow
        xi = alpha_dd * (prev_amp ** 2 / grund_pow) \
             + (1 - alpha_dd) * np.maximum(gamma - 1, 0)
        xi = np.maximum(xi, xi_min)

        v = xi / (1 + xi) * gamma
        gain = (np.sqrt(np.pi) / 2) * (np.sqrt(v) / np.maximum(gamma, EPS)) \
               * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
        gain = np.minimum(gain, 1.0)

        Hs_mag[:, [k]] = gain * Y_mag[:, [k]]
        prev_amp = Hs_mag[:, [k]]

    Ns_mag = Y_mag - Hs_mag
    return _istft(Ns_mag, Y_phase, sr, len(y)), _istft(Hs_mag, Y_phase, sr, len(y))


# ------------------------------------------------------- Struktur/Faktorisierung
def nmf(y, sr, K=8, n_iter=300, seed=0):
    """NMF mit KL-Divergenz. Zuordnung ueber die Aehnlichkeit zum Referenzspektrum."""
    f, t, V, Yph = _stft(y, sr)

    rng = np.random.default_rng(seed)
    m, n = V.shape
    W = rng.random((m, K)) + 1e-6
    H = rng.random((K, n)) + 1e-6
    ones = np.ones((m, n))

    for _ in range(n_iter):
        WH = W @ H + EPS
        H *= (W.T @ (V / WH)) / (W.T @ ones + EPS)
        WH = W @ H + EPS
        W *= ((V / WH) @ H.T) / (ones @ H.T + EPS)

    referenz = _grundspektrum(V).ravel()
    sim = _aehnlichkeit(W, referenz)
    ist_nutz = sim >= SIM_SCHWELLE
    if not ist_nutz.any():
        ist_nutz[np.argmax(sim)] = True

    WH_ges = W @ H + EPS
    Ns_mag = (W[:, ist_nutz] @ H[ist_nutz]) / WH_ges * V
    Hs_mag = (W[:, ~ist_nutz] @ H[~ist_nutz]) / WH_ges * V
    return _istft(Ns_mag, Yph, sr, len(y)), _istft(Hs_mag, Yph, sr, len(y))


def hpss(y, sr, kernel_zeit=51, kernel_freq=17, power=2.0):
    """HPSS (FitzGerald 2010). Harmonisch = Maschine, percussiv = Stoerschall."""
    f, t, Y_mag, Y_phase = _stft(y, sr)

    H_ref = median_filter(Y_mag, size=(1, kernel_zeit)) ** power
    P_ref = median_filter(Y_mag, size=(kernel_freq, 1)) ** power

    Ns_mag = Y_mag * (H_ref / (H_ref + P_ref + EPS))
    Hs_mag = Y_mag * (P_ref / (H_ref + P_ref + EPS))
    return _istft(Ns_mag, Y_phase, sr, len(y)), _istft(Hs_mag, Y_phase, sr, len(y))


def rpca(y, sr, lam=None, tol=1e-7, max_iter=200):
    """RPCA via Inexact ALM (Candes 2011). Low-rank = Maschine, sparse = Stoerschall."""
    f, t, M, Yph = _stft(y, sr)

    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    norm2 = np.linalg.norm(M, 2)
    normF = np.linalg.norm(M, "fro")
    mu = 1.25 / norm2
    mu_bar = mu * 1e7
    rho = 1.5

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Yd = M / max(norm2, np.linalg.norm(M, np.inf) / lam)

    for _ in range(max_iter):
        U, sig, Vt = np.linalg.svd(M - S + Yd / mu, full_matrices=False)
        L = (U * np.maximum(sig - 1 / mu, 0)) @ Vt
        Tm = M - L + Yd / mu
        S = np.sign(Tm) * np.maximum(np.abs(Tm) - lam / mu, 0)
        Z = M - L - S
        Yd = Yd + mu * Z
        mu = min(mu * rho, mu_bar)
        if np.linalg.norm(Z, "fro") / normF < tol:
            break

    return _istft(L, Yph, sr, len(y)), _istft(S, Yph, sr, len(y))


# ----------------------------------------------------------- adaptive Zerlegung
def emd(y, sr, max_imf=10):
    """EMD (Huang 1998). Zuordnung der IMFs ueber die Aehnlichkeit zum Referenzspektrum."""
    from PyEMD import EMD as _EMD

    imfs = _EMD().emd(y, max_imf=max_imf)

    _, _, Y_mag, _ = _stft(y, sr)
    referenz = _grundspektrum(Y_mag).ravel()
    spektren = np.column_stack([_mittelspektrum(imf, sr) for imf in imfs])
    sim = _aehnlichkeit(spektren, referenz)
    ist_nutz = sim >= SIM_SCHWELLE
    if not ist_nutz.any():
        ist_nutz[np.argmax(sim)] = True

    ns = imfs[ist_nutz].sum(axis=0)
    hs = imfs[~ist_nutz].sum(axis=0)
    ns += y - (ns + hs)   # Residuum (Trend) dem Nutzschall zuschlagen
    return ns, hs


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
