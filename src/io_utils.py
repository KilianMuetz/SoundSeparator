"""
io_utils.py — Gemeinsames Laden von Mischsignal und getrenntem Nutzsignal.

Vorher identisch dreimal in detect.py, fidelity.py und sisdr.py definiert.
"""

import soundfile as sf
from scipy.signal import resample_poly


def signal_laden(verfahren, eintrag, mix_dir, getrennt_dir, sr_ziel, sr_quelle):
    """Getrenntes Nutzsignal, fuer 'roh' das dezimierte Mischsignal."""
    if verfahren == "roh":
        y, sr = sf.read(mix_dir / eintrag["file"], dtype="float64")
        assert sr == sr_quelle, f"{eintrag['mix_id']}: Abtastrate {sr} != {sr_quelle}"
        return resample_poly(y, sr_ziel // 100, sr // 100)
    y, _ = sf.read(getrennt_dir / verfahren / f"{eintrag['mix_id']}_Ns.wav",
                   dtype="float64")
    return y
