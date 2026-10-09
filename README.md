# SoundSeparator

Code zur Masterarbeit *Gewinnung von Nutzschall in komplexen Geräuschumgebungen*
von Kilian Mütz.

Acht Verfahren der einkanaligen Signaltrennung werden darauf geprüft, wie gut
sie Maschinenschall von Störschall trennen. Bewertet wird, ob ein
Isolation Forest auf dem getrennten Signal Anomalien eines Ventilators
zuverlässiger erkennt als auf dem unbearbeiteten Mischsignal.

## Installation

Python 3.11 oder neuer.

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux, macOS
pip install -r requirements.txt
```

## Ablauf

Die Skripte laufen nacheinander aus dem Projektordner.

| Schritt | Aufruf | Ergebnis |
|---|---|---|
| 1 | `python src/mix_signals.py` | 228 Mischsignale in `data/mixed` |
| 2 | `python src/apply_methods.py` | getrennte Signale in `ergebnisse/getrennt`, Rechenzeiten |
| 3 | `python src/detect.py` | AUC und Konfusionsmatrix je Verfahren |
| 4 | `python src/sisdr.py` | Trennqualität als SI-SDR |
| 5 | `python src/abbildungen.py` | Spektrogramme und Konfusionsmatrizen |

`python src/trennschaerfe.py` bewertet die zehn Prototypen aus Kapitel 3 am
Feldsignal `data/observ_1.wav`.

Schritt 2 dauert am längsten und nutzt alle Prozessorkerne.

## Dateien

| Datei | Inhalt |
|---|---|
| `src/mix_signals.py` | Mischt Nutz- und Störschall, gleich laut, Störschall ab Sekunde 2 |
| `src/verfahren_lib.py` | Die Trennverfahren |
| `src/daten.py` | Pfade, Aufteilung in Training und Test, Laden der Signale |
| `src/apply_methods.py` | Wendet alle Verfahren auf alle Mischsignale an |
| `src/detect.py` | Anomalieerkennung mit Isolation Forest |
| `src/sisdr.py` | Trennqualität gegen die Ground Truth |
| `src/abbildungen.py` | Abbildungen für Kapitel 6 |
| `src/trennschaerfe.py` | Trennschärfe der Prototypen |

## Daten

`data/set` enthält die Rohaufnahmen.

- **Nutzschall:** Ventilator im Normalzustand und mit drei Anomalien in je zwei
  Schweregraden, aufgenommen im Freifeldraum.
- **Störschall:** eigene Aufnahmen und Ausschnitte aus dem Datensatz ESC-50
  (K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*,
  ACM Multimedia 2015). Es gelten die Lizenzen der einzelnen ESC-50-Clips.

Die Aufnahmen entstanden im Rahmen des Projekts SIPREMA bei Röwaplan.

## Ergebnisse

Alle Tabellen der Arbeit stammen aus den CSV-Dateien in `ergebnisse/protokoll`.
Die Mischung ist deterministisch, die Erkennung nutzt die festen Startwerte 0 bis 4.
