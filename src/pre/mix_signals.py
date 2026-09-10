"""
mix_signals.py — Erzeugt Mischsignale aus Nutzschall- und Stoerquellen-Aufnahmen.

Zwei Quelltypen:
  - "continuous": Hintergrundgeraeusch, laeuft ab ONSET_S bis zum Segmentende.
  - "transient":  Kurzes Einzelereignis, wird bei ONSET_S eingesetzt
                  (Rest bleibt Nutzschall pur).

Je Kombination werden mehrere Segmente aus unterschiedlichen Abschnitten
derselben Aufnahme gebildet. Die Normalsegmente teilen sich in einen
Trainings- und einen zeitlich davon getrennten Testbereich der Aufnahme
(Feld "rolle"), damit die Anomalieerkennung nicht auf demselben
Aufnahmeabschnitt trainiert und bewertet wird.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import json

TARGET_DURATION = 5.0     # Segmentlaenge des SIPREMA-Systems
TARGET_PEAK = 0.9         # normalisierter Ziel-Peak (Clipping-Puffer)
ONSET_S = 2.0             # Einsatz des Stoerschalls innerhalb des Segments
SNR_DB = 0.0              # Ziel-SNR Nutzschall vs. Stoerquelle (RMS-basiert)

ANOMALIE_OFFSETS = [0.0, 5.0]          # zwei Segmente je Anomalieaufnahme
NORMAL_TRAIN_ANTEILE = [0.02, 0.14, 0.26, 0.38]   # erstes Drittel der Normalaufnahme
NORMAL_TEST_ANTEILE = [0.66, 0.80, 0.94]          # letztes Drittel, klar getrennt

NUTZSCHALL = {
    "normal":       ("normalzustand.wav", None),
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
    if peak > TARGET_PEAK:
        mix = mix / peak * TARGET_PEAK
    return mix


def build_mix(nutz_path, stoer_path, stoer_type, offset=0.0, sr=44100):
    nutz = load_mono(nutz_path, sr)
    nutz = trim_window(nutz, sr, TARGET_DURATION, offset=offset)

    stoer_full = load_mono(stoer_path, sr)
    stoer = np.zeros(int(TARGET_DURATION * sr))
    onset_n = int(ONSET_S * sr)
    rest_n = len(stoer) - onset_n

    if stoer_type == "continuous":
        # Offset zyklisch in die verfuegbare Laenge der Stoeraufnahme falten
        nutzbar = len(stoer_full) / sr - rest_n / sr
        stoer_offset = offset % nutzbar if nutzbar > 0 else 0.0
        stoer[onset_n:] = trim_window(stoer_full, sr, rest_n / sr, offset=stoer_offset)
    else:  # transient: kurzes Einzelereignis, Rest bleibt Nutzschall pur
        event_n = min(len(stoer_full), rest_n)
        stoer[onset_n:onset_n + event_n] = stoer_full[:event_n]

    return mix_at_snr(nutz, stoer, SNR_DB)


def main(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    manifest = []
    mix_id = 1

    # Offsets relativ zur tatsaechlichen Laenge der Normalaufnahme
    spanne = sf.info(data_dir / NUTZSCHALL["normal"][0]).duration - TARGET_DURATION
    normal_train = [spanne * f for f in NORMAL_TRAIN_ANTEILE]
    normal_test = [spanne * f for f in NORMAL_TEST_ANTEILE]
    print(f"Normalaufnahme: {spanne + TARGET_DURATION:.0f}s, "
          f"Training bis {normal_train[-1]:.0f}s, Test ab {normal_test[0]:.0f}s")

    for nutz_name, (nutz_file, _) in NUTZSCHALL.items():
        if nutz_name == "normal":
            plan = ([(o, "train") for o in normal_train]
                    + [(o, "test") for o in normal_test])
        else:
            plan = [(o, "test") for o in ANOMALIE_OFFSETS]

        for stoer_name, (stoer_file, stoer_type) in STOERQUELLEN.items():
            for seg, (offset, rolle) in enumerate(plan, 1):
                mix_name = f"M{mix_id:03d}_{nutz_name}_{stoer_name}_s{seg}.wav"
                mix = build_mix(data_dir / nutz_file, data_dir / stoer_file,
                                stoer_type, offset=offset)
                sf.write(out_dir / mix_name, mix, 44100, subtype="PCM_24")
                manifest.append({
                    "mix_id": f"M{mix_id:03d}",
                    "nutzschall": nutz_name,
                    "stoerquelle": stoer_name,
                    "segment": seg,
                    "offset_s": round(offset, 2),
                    "rolle": rolle,
                    "snr_db": SNR_DB,
                    "file": mix_name,
                })
                mix_id += 1

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{mix_id - 1} Mischsignale erzeugt.")


if __name__ == "__main__":
    main(data_dir="data/set", out_dir="data/mixed")