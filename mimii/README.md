# SipremaSoundSeperator

Trennstufe für SIPREMA. Sie trennt Maschinenschall von Störschall, bevor die
Merkmale berechnet werden. Grundlage ist die Masterarbeit *Gewinnung von
Nutzschall in komplexen Geräuschumgebungen*.

Internes Repository von Röwaplan. Nicht öffentlich.

## Aufbau

| Datei | Inhalt |
|---|---|
| `src/trennung.py` | Trennverfahren für die Box, nur numpy und scipy |

## Daten

Nicht im Repository.

- `NOISE/` Feldsegmente von SIPREMA-Boxen, je Fall Störsegmente und Normalzustand
- `data/mimii/` MIMII-Datensatz (Purohit et al. 2019)
