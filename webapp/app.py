"""
app.py — Lokale WebApp zur Nutzschall-Trennung.

Start:  python webapp/app.py   (aus dem Projektordner, venv aktiv)
Dann:   http://127.0.0.1:8000  oeffnet sich im Browser.

Der Browser schickt die Audiodatei roh im Body, Verfahren und Parameter
im Query-String. Zurueck kommt JSON mit Kennzahlen, Audio, Huellkurve
und Spektrogramm fuer Mischsignal, Nutzschall und Stoerschall.

Nur Python-Standardbibliothek plus die Pakete aus requirements.txt.
"""

import base64
import io
import json
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))
import verfahren_lib as vl

HOST, PORT = "127.0.0.1", 8000
SR_SIPREMA = 16000    # Abtastrate von SIPREMA, wie in daten.py
SPALTEN = 1200        # Aufloesung von Huellkurve und Spektrogramm in Pixeln
DYNAMIK_DB = 80       # Spektrogramm: alles unter Maximum minus 80 dB ist schwarz

# Name -> (Anzeigename, Funktion, Parametername, Standardwert aus der Thesis)
VERFAHREN = {
    "spectralsubtraction": ("Spektralsubtraktion", vl.spektralsubtraktion, "beta", 0.04),
    "wiener":              ("Wiener-Filter",       vl.wiener,              "alpha_dd", 0.98),
    "mmse_stsa":           ("MMSE-STSA",           vl.mmse_stsa,           "alpha_dd", 0.98),
}


def huellkurve(x):
    """Minimum und Maximum je Pixelspalte, damit der Browser nicht jeden Abtastwert zeichnet."""
    n = len(x) // SPALTEN * SPALTEN
    if n == 0:
        return x.tolist(), x.tolist()
    bloecke = x[:n].reshape(SPALTEN, -1)
    return bloecke.min(axis=1).round(4).tolist(), bloecke.max(axis=1).round(4).tolist()


def spektrogramm_db(x, sr):
    """Betragsspektrogramm in dB auf dem STFT-Raster der Verfahren, zeitlich auf SPALTEN verdichtet."""
    _, _, X = stft(x, fs=sr, nperseg=vl.NPERSEG, noverlap=vl.NOVERLAP)
    S = np.abs(X)
    if S.shape[1] > SPALTEN:
        n = S.shape[1] // SPALTEN * SPALTEN
        S = S[:, :n].reshape(S.shape[0], SPALTEN, -1).max(axis=2)
    return 20 * np.log10(S + vl.EPS)


def png(S_db, vmax):
    puffer = io.BytesIO()
    plt.imsave(puffer, S_db, vmin=vmax - DYNAMIK_DB, vmax=vmax,
               cmap="magma", origin="lower", format="png")
    return base64.b64encode(puffer.getvalue()).decode()


def wav(x, sr, skala):
    """16-bit-WAV. Alle drei Signale teilen dieselbe Skala, die Lautstaerken bleiben vergleichbar."""
    puffer = io.BytesIO()
    sf.write(puffer, np.clip(x / skala, -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(puffer.getvalue()).decode()


def zahl(v):
    """JSON kennt kein NaN."""
    return None if not np.isfinite(v) else float(v)


def trenne(daten, verfahren, parameter, auf_16khz):
    y, sr_ein = sf.read(io.BytesIO(daten), dtype="float64", always_2d=True)
    y = y.mean(axis=1)                       # Mono wie in mix_signals.load_mono
    sr = sr_ein
    if auf_16khz and sr != SR_SIPREMA:
        g = gcd(SR_SIPREMA, sr)
        y = resample_poly(y, SR_SIPREMA // g, sr // g)
        sr = SR_SIPREMA

    name, funktion, pname, standard = VERFAHREN[verfahren]
    wert = standard if parameter is None else parameter

    start = time.perf_counter()
    ns, hs = funktion(y, sr, **{pname: wert})   # nur die Trennung wird gemessen, wie apply_methods.py
    dauer = time.perf_counter() - start

    dauer_signal = len(y) / sr
    e_ns, e_hs = np.sum(ns ** 2), np.sum(hs ** 2)
    kennzahlen = {
        "verfahren": name,
        "parameter": f"{pname} = {wert:g}",
        "dauer_s": dauer,
        "signal_s": dauer_signal,
        "echtzeitfaktor": dauer_signal / max(dauer, 1e-9),
        "sr_ein": sr_ein,
        "sr": sr,
        "eps_rek": zahl(np.linalg.norm(y - (ns + hs)) / (np.linalg.norm(y) + vl.EPS)),
        "U": zahl(1 - abs(np.corrcoef(ns, hs)[0, 1])),
        "anteil_ns": zahl(e_ns / (e_ns + e_hs + vl.EPS)),
    }

    signale = {"mix": y, "ns": ns, "hs": hs}
    spektren = {k: spektrogramm_db(x, sr) for k, x in signale.items()}
    vmax = max(S.max() for S in spektren.values())   # gemeinsame Farbskala
    spitze = max(np.max(np.abs(y)), np.max(np.abs(ns)), np.max(np.abs(hs)), vl.EPS)
    skala = max(spitze, 1.0)   # Audio nur bei Uebersteuerung leiser, sonst Originalpegel

    ausgabe = {"kennzahlen": kennzahlen, "signale": {}}
    for k, x in signale.items():
        lo, hi = huellkurve(x / spitze)   # Anzeige: lautestes Signal fuellt die Hoehe
        ausgabe["signale"][k] = {"wav": wav(x, sr, skala), "lo": lo, "hi": hi,
                                 "spektrogramm": png(spektren[k], vmax),
                                 "rms_db": zahl(20 * np.log10(np.sqrt(np.mean(x ** 2)) + vl.EPS))}
    return ausgabe


class Handler(BaseHTTPRequestHandler):
    def senden(self, code, inhalt, typ):
        self.send_response(code)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(inhalt)))
        self.send_header("Cache-Control", "no-store")   # Browser zeigt immer die aktuelle Version
        self.end_headers()
        self.wfile.write(inhalt)

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self.senden(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/verfahren":
            liste = [{"id": k, "name": v[0], "param": v[2], "standard": v[3]} for k, v in VERFAHREN.items()]
            self.senden(200, json.dumps(liste).encode(), "application/json")
        else:
            self.senden(404, b"nicht gefunden", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/trennen":
            return self.senden(404, b"nicht gefunden", "text/plain")
        q = parse_qs(url.query)
        try:
            verfahren = q["verfahren"][0]
            parameter = float(q["param"][0]) if q.get("param", [""])[0] else None
            auf_16khz = q.get("sr16", ["1"])[0] == "1"
            daten = self.rfile.read(int(self.headers["Content-Length"]))
            ergebnis = trenne(daten, verfahren, parameter, auf_16khz)
            self.senden(200, json.dumps(ergebnis).encode(), "application/json")
        except Exception as fehler:   # Fehlermeldung landet im Statusbalken
            self.senden(400, json.dumps({"fehler": f"{type(fehler).__name__}: {fehler}"}).encode(),
                        "application/json")

    def log_message(self, *args):
        pass   # Konsole ruhig halten


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"SoundSeparator laeuft auf http://{HOST}:{PORT}  (Strg+C beendet)")
    webbrowser.open(f"http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Beendet.")
