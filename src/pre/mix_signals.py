"""
mix_signals.py — Erzeugt Mischsignale aus Nutzschall- und Stoerquellen-Aufnahmen.

Zwei Quelltypen:
  - "continuous": Hintergrundgeraeusch, wird auf TARGET_DURATION getrimmt.
  - "transient":  Kurzes Einzelereignis (~5s), wird bei ONSET_S in die
                  TARGET_DURATION eingesetzt (Rest bleibt Nutzschall pur).
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import json

TARGET_DURATION = 5.0   # Sekunden, durch kuerzeste Anomalie-Aufnahme (11.9s) begrenzt
TARGET_PEAK = 0.9  # normalisierter Ziel-Peak (relative Amplitude, <1.0 als Clipping-Puffer)
ONSET_S = 2.0             # Startzeit transienter Events innerhalb des Mix
SNR_DB = 0.0              # Ziel-SNR Nutzschall vs. Stoerquelle (RMS-basiert)

def load_mono(path, sr_target=44100):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == sr_target, f"{path}: Samplerate {sr} != {sr_target}"
    audio = audio.astype(np.float64)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * TARGET_PEAK

    return audio

NUTZSCHALL = {
    "normal":       ("normalzustand.wav", None),        # Offset wird zufaellig/fix gewaehlt
    "a1_leicht":    ("gewicht_leicht.wav", 0.0),
    "a1_stark":     ("gewicht_schwer.wav", 0.0),
    "a2_leicht":    ("locker_leicht.wav", 0.0),
    "a2_deutlich":  ("locker_schwer.wav", 0.0),
    "a3_leicht":    ("tape_leicht.wav", 0.0),
    "a3_deutlich":  ("tape_schwer.wav", 0.0),
}

STOERQUELLEN = {
    "tuerschlaege": ("tuerschlaege.wav", "continuous"),
    "metallhammer": ("metallhammer.wav", "continuous"),
    "gespraech":    ("gespraech_6dzb.wav", "continuous"),
    "kraehe":       ("kraehe.wav", "transient"),
    "voegel":       ("voegel.wav", "transient"),
    "kuh":          ("kuh.wav", "transient"),
    "alarm":        ("alarm.wav", "transient"),
    "hupe":         ("hupe.wav", "transient"),
    "radio":        ("radio_6dzb.wav", "continuous"),
    "verkehr":      ("verkehr.wav", "continuous"),
    "akku_konstant":("makita_konstant.wav", "continuous"),
    "akku_dynamisch":("makita_dynamisch.wav", "continuous"),
}


def load_mono(path, sr_target=44100):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == sr_target, f"{path}: Samplerate {sr} != {sr_target}"
    return audio.astype(np.float64)


def trim_window(audio, sr, duration, offset=0.0):
    n = int(duration * sr)
    start = int(offset * sr)
    end = start + n
    if end > len(audio):
        raise ValueError(f"Aufnahme zu kurz fuer {duration}s ab Offset {offset}s")
    return audio[start:end]


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_at_snr(nutzschall, stoer, snr_db):
    target_stoer_rms = rms(nutzschall) / (10 ** (snr_db / 20))
    stoer_scaled = stoer * (target_stoer_rms / rms(stoer))
    mix = nutzschall + stoer_scaled
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        mix = mix / peak * 0.99
    return mix


def build_mix(nutz_path, stoer_path, stoer_type, offset=0.0, sr=44100):
    nutz = load_mono(nutz_path, sr)
    nutz = trim_window(nutz, sr, TARGET_DURATION, offset=offset)

    stoer_full = load_mono(stoer_path, sr)
    stoer = np.zeros(int(TARGET_DURATION * sr))
    onset_n = int(ONSET_S * sr)
    rest_n = len(stoer) - onset_n

    if stoer_type == "continuous":
        # Hintergrund ebenfalls versetzt, damit sich die Segmente unterscheiden
        stoer[onset_n:] = trim_window(stoer_full, sr, rest_n / sr, offset=offset)
    else:  # transient: kurzes Einzelereignis, Rest bleibt Nutzschall pur
        event_n = min(len(stoer_full), rest_n)
        stoer[onset_n:onset_n + event_n] = stoer_full[:event_n]

    return mix_at_snr(nutz, stoer, SNR_DB)


OFFSETS = [0.0, 5.0]   # zwei Segmente je Kombination aus derselben Aufnahme


def main(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    manifest = []
    mix_id = 1

    for nutz_name, (nutz_file, _) in NUTZSCHALL.items():
        for stoer_name, (stoer_file, stoer_type) in STOERQUELLEN.items():
            for seg, offset in enumerate(OFFSETS, 1):
                mix_name = f"M{mix_id:03d}_{nutz_name}_{stoer_name}_s{seg}.wav"
                mix = build_mix(data_dir / nutz_file, data_dir / stoer_file,
                                stoer_type, offset=offset)
                sf.write(out_dir / mix_name, mix, 44100, subtype="PCM_24")
                manifest.append({
                    "mix_id": f"M{mix_id:03d}",
                    "nutzschall": nutz_name,
                    "stoerquelle": stoer_name,
                    "segment": seg,
                    "snr_db": SNR_DB,
                    "file": mix_name,
                })
                mix_id += 1

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{mix_id - 1} Mischsignale erzeugt.")


if __name__ == "__main__":
    main(data_dir="data/set", out_dir="data/mixed")