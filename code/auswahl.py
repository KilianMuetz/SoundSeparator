"""
Auswertung der Zerlegungsguete (Option 1) je Verfahren am Referenzfile.

Struktur:  sep/<methode>/<methode>Ns.wav  und  sep/<methode>/<methode>Hs.wav
Das Skript durchsucht sep/ automatisch nach allen Methoden-Unterordnern.

Bewertung gegen das Original  data/observ_1.wav:
  eps_rek =  || y - (Ns + Hs) ||  /  || y ||          (Rekonstruktionsfehler)
  rho     =  | corr(Ns, Hs) |                          (Komplementaritaet)

Ein Verfahren trennt umso besser, je kleiner eps_rek (nichts verloren/erfunden)
UND je kleiner rho (Ns und Hs enthalten unterschiedliche Anteile).
Klassensieger = kleinstes rho innerhalb der Klasse bei eps_rek <= toleranz.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

BASE = Path(__file__).parent.parent   # Skript liegt in code/, data+sep im Root
sep_dir = BASE / "sep"

toleranz = 0.05   # eps_rek-Schwelle: oberhalb gilt die Rekonstruktion als verletzt

# Original laden (Referenz fuer den Rekonstruktionsfehler)
y, sr = sf.read(BASE / "data" / "observ_1.wav")
if y.ndim > 1:
    y = y.mean(axis=1)   # auf mono mitteln, falls noetig
y = y.astype(np.float64)

# Klassenzuordnung je Ordnername. Ordner, die hier nicht auftauchen, werden
# unter "Sonstige/ohne Klasse" gefuehrt (z.B. demucs, explorative).
klasse_von = {
    "synchronous_averaging": "Physikalisch",
    "cyclostationary":       "Physikalisch",
    "combfilter":            "Physikalisch",
    "cepstrum":              "Physikalisch",

    "spectralsubtraction":   "Statistisch (Spektral)",
    "mmse_stsa":             "Statistisch (Spektral)",
    "wiener":                "Statistisch (Spektral)",

    "kalman_harmonic":       "Statistisch (modellbasiert)",
    "kalman_ar":             "Statistisch (modellbasiert)",
    "statespace_em":         "Statistisch (modellbasiert)",

    "anc":                   "Adaptive Filter",
    "rls":                   "Adaptive Filter",

    "hpss":                  "Struktur/Faktorisierung",
    "nmf":                   "Struktur/Faktorisierung",
    "rpca":                  "Struktur/Faktorisierung",

    "vmd":                   "Modenzerlegung",
    "emd":                   "Modenzerlegung",
    "eemd":                  "Modenzerlegung",
    "ceemdan":               "Modenzerlegung",
    "ssa":                   "Modenzerlegung",
    "matching_pursuit":      "Modenzerlegung",

    "wavelet":               "Zeit-Frequenz/Sparse",
    "spectral_kurtosis":     "Zeit-Frequenz/Sparse",

    "konturierung":          "Psychoakustisch",

    "demucs":                "DL-Referenz",
}

# Ordner, die keine bewertbaren Trennverfahren sind -> ueberspringen
ignorieren = {"explorative"}

# Alle Methoden-Unterordner automatisch einsammeln
methoden_ordner = sorted(
    p.name for p in sep_dir.iterdir()
    if p.is_dir() and p.name not in ignorieren
)

ergebnisse = []   # (klasse, name, eps_rek, rho)

for name in methoden_ordner:
    ordner = sep_dir / name
    pfad_ns = ordner / f"{name}Ns.wav"
    pfad_hs = ordner / f"{name}Hs.wav"

    if not pfad_ns.exists() or not pfad_hs.exists():
        print(f"[uebersprungen] {name:24s} - Datei fehlt "
              f"({pfad_ns.name} / {pfad_hs.name})")
        continue

    ns, _ = sf.read(pfad_ns)
    hs, _ = sf.read(pfad_hs)
    if ns.ndim > 1:
        ns = ns.mean(axis=1)
    if hs.ndim > 1:
        hs = hs.mean(axis=1)
    ns = ns.astype(np.float64)
    hs = hs.astype(np.float64)

    # Auf gemeinsame Minimallaenge zuschneiden (Kachelung/Blockung kann
    # leicht abweichende Laengen erzeugen)
    n = min(len(y), len(ns), len(hs))
    y_n, ns_n, hs_n = y[:n], ns[:n], hs[:n]

    # Rekonstruktionsfehler
    nenner_rek = np.linalg.norm(y_n)
    eps_rek = np.linalg.norm(y_n - (ns_n + hs_n)) / nenner_rek if nenner_rek > 0 else np.nan

    # Komplementaritaet: Betrag des Korrelationskoeffizienten zwischen Ns und Hs
    ns_c = ns_n - ns_n.mean()
    hs_c = hs_n - hs_n.mean()
    nenner_rho = np.linalg.norm(ns_c) * np.linalg.norm(hs_c)
    rho = abs(np.dot(ns_c, hs_c) / nenner_rho) if nenner_rho > 0 else np.nan

    klasse = klasse_von.get(name, "Sonstige/ohne Klasse")
    ergebnisse.append((klasse, name, eps_rek, rho))

# --- Ausgabe: nach Klasse gruppiert, Sieger je Klasse markiert ---
print("\n" + "=" * 66)
print(f"{'Verfahren':24s} {'eps_rek':>9s} {'rho':>8s}   Sieger/Hinweis")
print("-" * 66)

# Klassenreihenfolge festlegen (Definitionsreihenfolge + Rest)
reihenfolge = []
for k in list(klasse_von.values()) + ["Sonstige/ohne Klasse"]:
    if k not in reihenfolge:
        reihenfolge.append(k)
vorhandene_klassen = [k for k in reihenfolge if any(e[0] == k for e in ergebnisse)]

for klasse in vorhandene_klassen:
    zeilen = [e for e in ergebnisse if e[0] == klasse]

    # Sieger = kleinstes rho unter denen, die die Rekonstruktion einhalten
    gueltige = [e for e in zeilen if e[2] <= toleranz]
    sieger_name = min(gueltige, key=lambda e: e[3])[1] if gueltige else None

    print(f"\n[{klasse}]")
    for _, name, eps_rek, rho in sorted(zeilen, key=lambda e: e[3]):
        mark = "  <-- Sieger" if name == sieger_name else ""
        warn = "  (!) Rekonstruktion verletzt" if eps_rek > toleranz else ""
        print(f"{name:24s} {eps_rek:9.4f} {rho:8.4f}{mark}{warn}")

print("\n" + "=" * 66)
print(f"Toleranz eps_rek <= {toleranz}. "
      f"Sieger = kleinstes rho bei eingehaltener Rekonstruktion.")
print("Hinweis: Differenz-basierte Verfahren (Hs = y - Ns) haben eps_rek ~ 0")
print("per Konstruktion; dort ist rho die massgebliche Kennzahl.")