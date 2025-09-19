# lidar_fit_tool

Programm zum Kalibrieren von Temperatur- und Wasserdampf-daten aus preparierten NetCDF-Lidar Dateien.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/17f9a6fe-f43c-454b-9244-946903c1847d" />


## Description

Gui mit 3 Tabs:

Wasserdampf: Q_ana = WV/RR1 wird berechnet und mit T = Q_ana * A an das Mixing-Ratio[g/kg] einer Radiosonde gefittet.
    Overlap-korrektur wird anschließend angewendet als Q_korr = Q_ana * A + OV_korr

Temperatur: Q_ana = RR2/RR1 wird berechnet. Mit Q = a + b / T + c / (T ** 2) wird T der Radiosonde an Q gefittet (?). 
    Overlap-korrektur wird anschließend angewendet als Q_korr = Q_ana + OV_korr . Dann Rückrechnung T(Q) = 1 / ( -(b / (2 * c)) + sqrt( (b / (2 * c))^2 - (a - ln(Q_korr)) / c ) )

Radiosonde:
    nimmt Station id, Datum, Zeit in UTC, und speicherort und lädt dann Sounding-daten von UWYO als csv herunter. Der Download kann einige Sekunden dauern.
    Vorsicht: Die website lädt bei falschen UTC-Zeiten die nächstmögliche Messung herunter. Zur kontrolle am besten Startzeit in der CSV Überprüfen.

Generelles:
    Plotfenster können über die Navigationtoolbar geändert werden. Durch klick auf achsenbeschriftung kann das intervall festgelegt werden, aber nur, wenn nichts aus der Navigationtoolbar ausgewählt ist. 

## Getting Started

Mit Conda Umgebung:

conda env create -f environment.yml

conda activate lidar-fit

### Dependencies

pip install -r requirements.txt

### Executing program

python main_gui.py

## Author

David Ritter

david.ritter001@gmail.com


