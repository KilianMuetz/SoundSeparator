"""
trennung.py — Trennstufe fuer SIPREMA.

Jede Funktion hat die Signatur  trenne(y, sr, referenz=None) -> (ns, hs)
  ns = geschaetzter Nutzschall  (Maschine)
  hs = geschaetzter Stoerschall (Rest)
Es gilt stets  ns + hs == y.

referenz=None schaetzt das Referenzspektrum aus dem Segment selbst,
wie in der Masterarbeit. Wird eine Referenz uebergeben, etwa fortgeschrieben
ueber viele Segmente, wird diese genutzt.

Spektralsubtraktion, Wiener und spek_wiener nur mit numpy und scipy, damit sie auf der Box laufen.
HPSS braucht librosa, das auf der Box ebenfalls installiert ist.
"""

import librosa
import numpy as np
from scipy.signal import istft, stft

NPERSEG = 1024
NOVERLAP = 512
PERZENTIL = 20
EPS = 1e-12


def spektrogramm(y, sr):
    _, _, Y = stft(y, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return np.abs(Y), np.angle(Y)


def zeitsignal(betrag, phase, sr, n):
    _, x = istft(betrag * np.exp(1j * phase), fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    x = x[:n]
    return np.concatenate([x, np.zeros(n - len(x))])


def referenzspektrum(Y_mag, perzentil=PERZENTIL):
    """Was dauerhaft im Segment steht, je Frequenz (Stahl et al. 2000)."""
    return np.percentile(Y_mag, perzentil, axis=1, keepdims=True)


def _aufteilen(y, sr, Ns_mag, phase):
    ns = zeitsignal(Ns_mag, phase, sr, len(y))
    return ns, y - ns


def spektralsubtraktion(y, sr, referenz=None, perzentil=PERZENTIL, beta=0.04):
    """Spektralsubtraktion (Boll 1979), wie in der Masterarbeit.

    Harte Klemme: Ns = min(R, (1 - beta) * Y).
    """
    Y_mag, phase = spektrogramm(y, sr)
    R = referenzspektrum(Y_mag, perzentil) if referenz is None else referenz
    Hs_mag = np.maximum(Y_mag - R, beta * Y_mag)
    return _aufteilen(y, sr, Y_mag - Hs_mag, phase)


def wiener(y, sr, referenz=None, perzentil=PERZENTIL, alpha_dd=0.98):
    """Wiener-Filter mit Decision-Directed-Schaetzung, wie in der Masterarbeit."""
    Y_mag, phase = spektrogramm(y, sr)
    R = referenzspektrum(Y_mag, perzentil) if referenz is None else referenz
    ref_pow = R ** 2 + EPS
    Hs_mag = np.zeros_like(Y_mag)
    vorher = Y_mag[:, [0]]
    for k in range(Y_mag.shape[1]):
        gamma = Y_mag[:, [k]] ** 2 / ref_pow
        xi = alpha_dd * vorher ** 2 / ref_pow + (1 - alpha_dd) * np.maximum(gamma - 1, 0)
        Hs_mag[:, [k]] = xi / (1 + xi) * Y_mag[:, [k]]
        vorher = Hs_mag[:, [k]]
    return _aufteilen(y, sr, Y_mag - Hs_mag, phase)


def spek_wiener(y, sr, referenz=None, perzentil=PERZENTIL, beta=0.04):
    """Weiche Spektralsubtraktion: Wiener-Gain aus den Schaetzungen der Spektralsubtraktion.

    Maschine  = R^2
    Stoerung  = max(Y^2 - R^2, beta^2 * Y^2)
    Gain      = Maschine / (Maschine + Stoerung)
    Statt einer harten Klemme wird jedes Feld weich gedaempft.
    """
    Y_mag, phase = spektrogramm(y, sr)
    R = referenzspektrum(Y_mag, perzentil) if referenz is None else referenz
    maschine = np.broadcast_to(R ** 2, Y_mag.shape)
    stoerung = np.maximum(Y_mag ** 2 - maschine, beta ** 2 * Y_mag ** 2)
    gain = maschine / (maschine + stoerung + EPS)
    return _aufteilen(y, sr, gain * Y_mag, phase)


def hpss(y, sr, referenz=None, filterlaenge=17, p=2.0):
    """HPSS (FitzGerald 2010), wie in der Masterarbeit. Harmonisch = Maschine, perkussiv = Stoerschall.

    Braucht kein Referenzspektrum, referenz wird ignoriert.
    """
    Y_mag, phase = spektrogramm(y, sr)
    maske_h, _ = librosa.decompose.hpss(Y_mag, kernel_size=filterlaenge, power=p, mask=True)
    return _aufteilen(y, sr, Y_mag * maske_h, phase)


VERFAHREN = {
    "spektralsubtraktion": spektralsubtraktion,
    "wiener": wiener,
    "spek_wiener": spek_wiener,
    "hpss": hpss,
}
