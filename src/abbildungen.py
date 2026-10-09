"""abbildungen.py — Spektrogramme und Konfusionsmatrizen fuer Kapitel 6."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from daten import BASE, PROTOKOLL, SR, manifest, nutzsignal

ZIEL = BASE / "ergebnisse" / "abbildungen"
NAMEN = {
    "roh": "Mischsignal", "ground_truth": "Ground Truth",
    "synchronous_averaging": "Synchrone Mittelung", "wiener": "Wiener-Filter",
    "spectralsubtraction": "Spektralsubtraktion", "mmse_stsa": "MMSE-STSA",
    "nmf": "NMF", "hpss": "HPSS", "rpca": "RPCA", "emd": "EMD",
}


def spektrogramme(mix_id, verfahren, titel, datei, spalten=2):
    """Mischsignal, Ground Truth und die gewaehlten Verfahren nebeneinander."""
    e = next(e for e in manifest if e["mix_id"] == mix_id)
    namen = ["roh", "ground_truth", *verfahren]
    spektren = [np.abs(stft(nutzsignal(v, e), fs=SR, nperseg=512)[2]) for v in namen]
    referenz = spektren[0].max()   # gemeinsame Referenz fuer alle Felder

    zeilen = -(-len(namen) // spalten)
    fig, axs = plt.subplots(zeilen, spalten, figsize=(3.2 * spalten + 1.5, 3 * zeilen),
                            sharex=True, sharey=True, squeeze=False)
    for ax, name, S in zip(axs.flat, namen, spektren):
        bild = ax.imshow(20 * np.log10(S / referenz + 1e-10), origin="lower", aspect="auto",
                         extent=[0, 5, 0, SR / 2000], vmin=-80, vmax=0, cmap="magma")
        ax.set_title(NAMEN[name])
    for ax in axs[:, 0]:
        ax.set_ylabel("Frequenz [kHz]")
    for ax in axs[-1]:
        ax.set_xlabel("Zeit [s]")
    fig.suptitle(titel)
    fig.colorbar(bild, ax=axs, label="Pegel [dB, gemeinsame Referenz]", shrink=0.8)
    fig.savefig(ZIEL / datei, dpi=200, bbox_inches="tight")
    plt.close(fig)


def konfusionsmatrizen(datei="konfusionsmatrizen.png"):
    """Eine 2x2-Matrix je Verfahren, sortiert nach AUC."""
    with open(PROTOKOLL / "erkennung_auc.csv", encoding="utf-8") as f:
        auc = {z["verfahren"]: float(z["auc"]) for z in csv.DictReader(f)}
    with open(PROTOKOLL / "erkennung_metriken.csv", encoding="utf-8") as f:
        zeilen = [z for z in csv.DictReader(f) if z["verfahren"] != "ground_truth"]
    zeilen.sort(key=lambda z: -auc[z["verfahren"]])

    fig, axs = plt.subplots(3, 3, figsize=(9.5, 9.5))
    for ax, z in zip(axs.flat, zeilen):
        M = np.array([[z["tp"], z["fp"]], [z["fn"], z["tn"]]], dtype=int)
        anteil = M / M.sum(axis=0)   # Anteil je tatsaechlichem Zustand
        ax.imshow(anteil, cmap="Blues", vmin=0, vmax=1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=15, fontweight="bold",
                        color="white" if anteil[i, j] > 0.62 else "#1a1a1a")
        ax.set_title(NAMEN[z["verfahren"]])
        ax.set_xticks([0, 1], ["anomal", "normal"])
        ax.set_yticks([0, 1], ["anomal", "normal"])
        ax.tick_params(length=0)
    fig.supxlabel("Tatsächlicher Zustand")
    fig.supylabel("Einstufung durch den Detektor")
    fig.tight_layout()
    fig.savefig(ZIEL / datei, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ZIEL.mkdir(parents=True, exist_ok=True)
    unwucht = "Unwucht stark mit Akkuschrauber konstant"
    spektrogramme("M129", ["wiener", "spectralsubtraction", "mmse_stsa", "nmf", "rpca", "hpss"],
                  f"Verfahren über der Referenz, {unwucht}", "spektrogramme_M129_besser.png", 4)
    spektrogramme("M129", ["synchronous_averaging", "emd"],
                  f"Verfahren unter der Referenz, {unwucht}", "spektrogramme_M129_schlechter.png")
    spektrogramme("M171", ["spectralsubtraction", "emd"],
                  "Lose Befestigung deutlich mit Hupe", "spektrogramme_M171.png")
    spektrogramme("M223", ["spectralsubtraction", "emd"],
                  "Anstreifen deutlich mit Verkehr", "spektrogramme_M223.png")
    konfusionsmatrizen()
    print(f"Abbildungen in {ZIEL}")
