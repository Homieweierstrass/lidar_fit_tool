import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
from datetime import datetime
from io import StringIO
import os

import water_vapor_fit as wv
import temperature_fit as temp

# Default-Dateipfade
# Default-Dateipfade 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NETCDF_PATH = os.path.join(BASE_DIR, "lidar/20240823_031504_to_20240823_032953_Allgl_900s_97m.nc")
DEFAULT_SONDE_PATH = os.path.join(BASE_DIR, "radiosonde/sounding_11120_20240823_02UTC.csv")


def download_radiosonde_csv(station_id: str, date: str, hour: str = "00"):
    dt = datetime.strptime(date, "%Y%m%d")
    datetime_str = dt.strftime(f"%Y-%m-%d {hour}:00:00")

    url = (
        "https://weather.uwyo.edu/wsgi/sounding"
        f"?datetime={datetime_str}"
        f"&id={station_id}&src=BUFR&type=TEXT:CSV"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text

    if "Can't get" in text:
        return None, None

    # CSV direkt lesen
    df = pd.read_csv(StringIO(text))
    return df, datetime_str

def parse_limits(entry_min, entry_max):
    try:
        vmin = float(entry_min.get().strip()) if entry_min.get().strip() else None
        vmax = float(entry_max.get().strip()) if entry_max.get().strip() else None
        return vmin, vmax
    except ValueError:
        return None, None


class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lidar-Radiosonden Kalibrierung")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.setup_water_vapor_tab()
        self.setup_temperature_tab()
        self.setup_radiosonde_tab()


    # Klick-Handler für Achsenlabels
    def on_label_click(self, ax, axis="x", canvas=None):
        if axis == "x":
            curr_min, curr_max = ax.get_xlim()
            label = "X-Achse"
        else:
            curr_min, curr_max = ax.get_ylim()
            label = "Y-Achse"

        try:
            new_min = simpledialog.askfloat(
                "Achsenlimit", f"{label} min (aktuell {curr_min:.2f}):", parent=self.root
            )
            new_max = simpledialog.askfloat(
                "Achsenlimit", f"{label} max (aktuell {curr_max:.2f}):", parent=self.root
            )
            if new_min is not None and new_max is not None:
                if axis == "x":
                    ax.set_xlim(new_min, new_max)
                else:
                    ax.set_ylim(new_min, new_max)
                canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Achsenlimit nicht setzen:\n{e}")

    # Wasserdampf-Tab
    def setup_water_vapor_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Wasserdampf")

        # Dateipfade
        paths_frame = ttk.LabelFrame(frame, text="Dateien")
        paths_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(paths_frame, text="NetCDF Pfad:").grid(row=0, column=0, sticky="w")
        self.wv_netcdf_entry = ttk.Entry(paths_frame, width=80)
        self.wv_netcdf_entry.insert(0, DEFAULT_NETCDF_PATH)
        self.wv_netcdf_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(paths_frame, text="Radiosonde Pfad:").grid(row=1, column=0, sticky="w")
        self.wv_sonde_entry = ttk.Entry(paths_frame, width=80)
        self.wv_sonde_entry.insert(0, DEFAULT_SONDE_PATH)
        self.wv_sonde_entry.grid(row=1, column=1, padx=5, pady=2)

        # Fit-Einstellungen
        fit_frame = ttk.LabelFrame(frame, text="Fit-Einstellungen")
        fit_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(fit_frame, text="Fit min [km]:").grid(row=0, column=0)
        self.wv_fit_min = ttk.Entry(fit_frame, width=8)
        self.wv_fit_min.insert(0, "0.3")
        self.wv_fit_min.grid(row=0, column=1)

        ttk.Label(fit_frame, text="Fit max [km]:").grid(row=0, column=2)
        self.wv_fit_max = ttk.Entry(fit_frame, width=8)
        self.wv_fit_max.insert(0, "3.0")
        self.wv_fit_max.grid(row=0, column=3)

        # Parameter: Eingabe + Fit-Ergebnisse
        params_frame = ttk.LabelFrame(frame, text="Fitparameter")
        params_frame.pack(fill="x", padx=5, pady=5)

        # Manuelle Eingabe
        ttk.Label(params_frame, text="A (manuell):").grid(row=0, column=0)
        self.wv_A = ttk.Entry(params_frame, width=10)
        self.wv_A.grid(row=0, column=1)

        ttk.Label(params_frame, text="corr (manuell):").grid(row=0, column=2)
        self.wv_corr = ttk.Entry(params_frame, width=10)
        self.wv_corr.grid(row=0, column=3)

        # Ergebnisfeld (readonly, kopierbar)
        ttk.Label(params_frame, text="A (Fit):").grid(row=1, column=0)
        self.wv_A_result = ttk.Entry(params_frame, width=12, state="readonly")
        self.wv_A_result.grid(row=1, column=1, padx=5)

        # Overlap
        overlap_frame = ttk.LabelFrame(frame, text="Overlap-Funktion")
        overlap_frame.pack(fill="x", padx=5, pady=5)

        self.wv_overlap = ttk.Entry(overlap_frame, width=80)
        self.wv_overlap.insert(0, "-np.exp(-((x-180)**2)/8000)/800")
        self.wv_overlap.pack(padx=5, pady=5)

        # Buttons und Status
        ttk.Button(frame, text="Fit starten", command=self.do_water_vapor_fit).pack(pady=5)
        self.wv_status = ttk.Label(frame, text="Bitte Pfade prüfen.", foreground="blue")
        self.wv_status.pack()

        # Matplotlib Canvas und Toolbar
        plot_frame = ttk.Frame(frame)
        plot_frame.pack(fill="both", expand=True)

        self.wv_fig = plt.figure(figsize=(18, 6))
        gs = self.wv_fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = self.wv_fig.add_subplot(gs[0, 0])
        ax2 = self.wv_fig.add_subplot(gs[0, 1])
        ax3 = self.wv_fig.add_subplot(gs[1, 1])

        self.wv_axes = [ax1, ax2, ax3]

        self.wv_canvas = FigureCanvasTkAgg(self.wv_fig, master=plot_frame)

        self.wv_toolbar = NavigationToolbar2Tk(self.wv_canvas, plot_frame)
        self.wv_toolbar.update()
        self.wv_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.wv_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Klickbare Labels
        for ax in self.wv_axes:
            ax.xaxis.label.set_picker(True)
            ax.yaxis.label.set_picker(True)

        self.wv_fig.canvas.mpl_connect(
            "pick_event",
            lambda event: self.on_pick_label(event, self.wv_axes, self.wv_canvas),
        )

    def on_pick_label(self, event, axes, canvas):
        for ax in axes:
            if event.artist == ax.xaxis.label:
                self.on_label_click(ax, "x", canvas)
            elif event.artist == ax.yaxis.label:
                self.on_label_click(ax, "y", canvas)

    def do_water_vapor_fit(self):
        try:
            # Parameter einlesen
            netcdf_file = self.wv_netcdf_entry.get()
            sonde_file = self.wv_sonde_entry.get()
            fit_min = float(self.wv_fit_min.get()) * 1000  # km → m
            fit_max = float(self.wv_fit_max.get()) * 1000
            fit_a = self.wv_A.get().strip()
            fit_a = float(fit_a) if fit_a else None

            overlap_expr = self.wv_overlap.get().strip()

            # Fit
            heights, q_ana, q_corr, mr_interp, H_fit, MR_fit_curve, residuals, A, corr = (
                wv.perform_fit(netcdf_file, sonde_file, fit_min, fit_max, fit_a, overlap_expr)
            )

            # Höhen in km
            heights_km = heights / 1000
            H_fit_km = H_fit / 1000

            for ax in self.wv_axes:
                ax.clear()

            # Maske für gültige Werte
            plot_mask = ~np.isnan(q_corr) & ~np.isnan(mr_interp)

            # Sortierte Radiosonde &undLidar über alle Höhe
            sort_idx_all = np.argsort(heights_km[plot_mask])
            heights_sorted = heights_km[plot_mask][sort_idx_all]
            mr_sorted = mr_interp[plot_mask][sort_idx_all]
            q_corr_sorted = q_corr[plot_mask][sort_idx_all]

            # Sortierte Fit-Daten im Fitbereich
            sort_idx_fit = np.argsort(H_fit_km)
            H_fit_sorted = H_fit_km[sort_idx_fit]
            MR_fit_curve_sorted = MR_fit_curve[sort_idx_fit]
            residuals_sorted = residuals[sort_idx_fit]

            MR_fit_all = wv.fit_func(q_ana, A)+corr
            MR_fit_all_sorted = MR_fit_all[plot_mask][sort_idx_all]

            #Plots
            # Radiosonde vs. Lidar mit Overlap
            self.wv_axes[0].plot(heights_sorted, mr_sorted, "k", label="Radiosonde")
            self.wv_axes[0].plot(H_fit_sorted, MR_fit_curve_sorted, color="dodgerblue", linewidth=3, alpha=0.7, label=f"Fit A={A:.6f}")

            self.wv_axes[0].set_xlabel("Höhe a.s.l. [km]")
            self.wv_axes[0].set_ylabel("Mixing Ratio [g/kg]")
            self.wv_axes[0].grid(True)
            self.wv_axes[0].legend()

            # Extrapolierter Fit vs. Radiosonde
            self.wv_axes[1].plot(heights_sorted, MR_fit_all_sorted, "r", alpha=0.8, label="Fit + Overlap")
            self.wv_axes[1].plot(heights_sorted, mr_sorted, "k", label="Radiosonde")
            self.wv_axes[1].set_xlabel("Höhe a.s.l. [km]")
            self.wv_axes[1].set_ylabel("Mixing Ratio [g/kg]")
            self.wv_axes[1].legend()
            self.wv_axes[1].grid(True)

            # Residuen
            self.wv_axes[2].plot(H_fit_sorted, residuals_sorted, "b", label="Residuen")
            self.wv_axes[2].axhline(0, color="gray", linestyle="--")
            self.wv_axes[2].set_xlabel("Höhe a.s.l. [km]")
            self.wv_axes[2].set_ylabel("Residuen [g/kg]")
            self.wv_axes[2].legend()
            self.wv_axes[2].grid(True)

            # Fitbereich markieren
            for ax in self.wv_axes:
                ax.axvspan(fit_min / 1000, fit_max / 1000, color="gray", alpha=0.2)

            self.wv_fig.tight_layout()
            self.wv_canvas.draw()
            # Fitparameter in readonly Feld schreiben
            self.wv_A_result.config(state="normal")
            self.wv_A_result.delete(0, tk.END)
            self.wv_A_result.insert(0, f"{A:.6f}")
            self.wv_A_result.config(state="readonly")

            self.wv_status.config(text="Fit durchgeführt", foreground="green")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.wv_status.config(text="Fehler beim Fit!", foreground="red")

    # Temperatur-Tab
    def setup_temperature_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Temperatur")

        # Dateipfade
        paths_frame = ttk.LabelFrame(frame, text="Dateien")
        paths_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(paths_frame, text="NetCDF Pfad:").grid(row=0, column=0, sticky="w")
        self.temp_netcdf_entry = ttk.Entry(paths_frame, width=80)
        self.temp_netcdf_entry.insert(0, DEFAULT_NETCDF_PATH)
        self.temp_netcdf_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(paths_frame, text="Radiosonde Pfad:").grid(row=1, column=0, sticky="w")
        self.temp_sonde_entry = ttk.Entry(paths_frame, width=80)
        self.temp_sonde_entry.insert(0, DEFAULT_SONDE_PATH)
        self.temp_sonde_entry.grid(row=1, column=1, padx=5, pady=2)

        # Fit-Einstellungen
        fit_frame = ttk.LabelFrame(frame, text="Fit-Einstellungen")
        fit_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(fit_frame, text="Fit min [km]:").grid(row=0, column=0)
        self.temp_fit_min = ttk.Entry(fit_frame, width=8)
        self.temp_fit_min.insert(0, "1.5")
        self.temp_fit_min.grid(row=0, column=1)
        ttk.Label(fit_frame, text="Fit max [km]:").grid(row=0, column=2)
        self.temp_fit_max = ttk.Entry(fit_frame, width=8)
        self.temp_fit_max.insert(0, "3.5")
        self.temp_fit_max.grid(row=0, column=3)

        # Parameter: Eingabe + Fit-Ergebnisse
        params_frame = ttk.LabelFrame(frame, text="Fitparameter")
        params_frame.pack(fill="x", padx=5, pady=5)

        # Manuelle Eingabe
        ttk.Label(params_frame, text="a (manuell):").grid(row=0, column=0)
        self.temp_a = ttk.Entry(params_frame, width=10)
        self.temp_a.grid(row=0, column=1)

        ttk.Label(params_frame, text="b (manuell):").grid(row=0, column=2)
        self.temp_b = ttk.Entry(params_frame, width=10)
        self.temp_b.grid(row=0, column=3)

        ttk.Label(params_frame, text="c (manuell):").grid(row=0, column=4)
        self.temp_c = ttk.Entry(params_frame, width=10)
        self.temp_c.grid(row=0, column=5)

        # Ergebnisfelder (readonly, kopierbar)
        ttk.Label(params_frame, text="a (Fit):").grid(row=1, column=0)
        self.temp_a_result = ttk.Entry(params_frame, width=12, state="readonly")
        self.temp_a_result.grid(row=1, column=1, padx=5)

        ttk.Label(params_frame, text="b (Fit):").grid(row=1, column=2)
        self.temp_b_result = ttk.Entry(params_frame, width=12, state="readonly")
        self.temp_b_result.grid(row=1, column=3, padx=5)

        ttk.Label(params_frame, text="c (Fit):").grid(row=1, column=4)
        self.temp_c_result = ttk.Entry(params_frame, width=12, state="readonly")
        self.temp_c_result.grid(row=1, column=5, padx=5)

        # Overlap
        overlap_frame = ttk.LabelFrame(frame, text="Overlap-Funktion")
        overlap_frame.pack(fill="x", padx=5, pady=5)
        self.temp_overlap = ttk.Entry(overlap_frame, width=80)
        self.temp_overlap.insert(0, "-np.exp(-((x-3000)**2)/300000)")
        self.temp_overlap.pack(padx=5, pady=5)

        # Buttons und Status
        ttk.Button(frame, text="Fit starten", command=self.do_temperature_fit).pack(pady=5)
        self.temp_status = ttk.Label(frame, text="Bitte Pfade prüfen.", foreground="blue")
        self.temp_status.pack()

        # Matplotlib Canvas und Toolbar
        plot_frame = ttk.Frame(frame)
        plot_frame.pack(fill="both", expand=True)

        self.temp_fig = plt.figure(figsize=(18, 6))
        gs = self.temp_fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = self.temp_fig.add_subplot(gs[0, 0])
        ax2 = self.temp_fig.add_subplot(gs[0, 1])
        ax3 = self.temp_fig.add_subplot(gs[1, 1])


        self.temp_axes = [ax1, ax2, ax3]

        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, master=plot_frame)

        self.temp_toolbar = NavigationToolbar2Tk(self.temp_canvas, plot_frame)
        self.temp_toolbar.update()
        self.temp_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.temp_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Klickbare Labels
        for ax in self.temp_axes:
            ax.xaxis.label.set_picker(True)
            ax.yaxis.label.set_picker(True)

        self.temp_fig.canvas.mpl_connect(
            "pick_event",
            lambda event: self.on_pick_label(event, self.temp_axes, self.temp_canvas),
        )

    def do_temperature_fit(self):
        try:
            netcdf_file = self.temp_netcdf_entry.get()
            sonde_file = self.temp_sonde_entry.get()
            fit_min = float(self.temp_fit_min.get()) * 1000  # km → m
            fit_max = float(self.temp_fit_max.get()) * 1000

            params = None
            if (
                self.temp_a.get().strip()
                and self.temp_b.get().strip()
                and self.temp_c.get().strip()
            ):
                params = (
                    float(self.temp_a.get()),
                    float(self.temp_b.get()),
                    float(self.temp_c.get()),
                )

            overlap_expr = self.temp_overlap.get().strip()

            (
                heights,
                q_ana,
                temp_interp,
                H_fit,
                Q_fit,
                T_fit_curve,
                residuals,
                fit_params,
                T_from_data_corr,
            ) = temp.perform_fit(
                netcdf_file,
                sonde_file,
                fit_min,
                fit_max,
                None,
                None,
                params,
                overlap_expr,
            )

            a, b, c = fit_params
            # Fitparameter in die readonly Felder schreiben
            for entry, val in zip(
                [self.temp_a_result, self.temp_b_result, self.temp_c_result],
                [a, b, c]
            ):
                entry.config(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, f"{val:.6f}")
                entry.config(state="readonly")


            n = min(len(heights), len(q_ana), len(temp_interp))
            heights, q_ana, temp_interp, T_from_data_corr = (
                heights[:n],
                q_ana[:n],
                temp_interp[:n],
                T_from_data_corr[:n],
            )

            heights_km = heights / 1000

            for ax in self.temp_axes:
                ax.clear()

            plot_mask = (
                ~np.isnan(q_ana) & ~np.isnan(temp_interp) & ~np.isnan(T_from_data_corr)
            )
            fit_mask_range = (
                (heights >= fit_min) & (heights <= fit_max) & plot_mask
            )

            #Plots
            # T vs Q_ana
            self.temp_axes[0].plot(
                q_ana[plot_mask], temp_interp[plot_mask], "k", label="Radiosonde"
            )
            T_grid = np.linspace(np.nanmin(temp_interp[plot_mask]),
                                np.nanmax(temp_interp[plot_mask]), 300)
            Q_forward = temp.forward_Q_from_T(T_grid, a, b, c)
            self.temp_axes[0].plot(
                Q_forward,
                T_grid,
                color="dodgerblue",
                linewidth=3,
                alpha=0.7,
                label=f"Fit: a={a:.6f}, b={b:.6f}, c={c:.6f}",
            )
            q_fit_min = np.nanmin(q_ana[fit_mask_range])
            q_fit_max = np.nanmax(q_ana[fit_mask_range])
            self.temp_axes[0].axvspan(q_fit_min, q_fit_max, color="gray", alpha=0.14)

            self.temp_axes[0].set_xlabel("Q_ana")
            self.temp_axes[0].set_ylabel("Temperatur [K]")
            self.temp_axes[0].legend()
            self.temp_axes[0].grid(True)

            # T vs. Height
            self.temp_axes[1].plot(
                heights_km[plot_mask], temp_interp[plot_mask], color="black", label="Radiosonde"
            )
            self.temp_axes[1].plot(
                heights_km[plot_mask],
                T_from_data_corr[plot_mask],
                color="red",
                alpha=0.8,
                label="Fit + Overlap",
            )
            self.temp_axes[1].set_xlabel("Höhe a.s.l. [km]")
            self.temp_axes[1].set_ylabel("Temperatur [K]")
            self.temp_axes[1].legend()
            self.temp_axes[1].grid(True)

            # Residuen
            residuals_corr = T_from_data_corr[plot_mask] - temp_interp[plot_mask]
            self.temp_axes[2].plot(
                heights_km[plot_mask], residuals_corr, color="blue", label="Residuen"
            )
            self.temp_axes[2].set_ylim(-2, 2)
            self.temp_axes[2].set_xlim(0, 12)  
            self.temp_axes[2].axhline(0, color="gray", linestyle="--")
            self.temp_axes[2].set_xlabel("Höhe a.s.l. [km]")
            self.temp_axes[2].set_ylabel("Residuen [K]")
            self.temp_axes[2].legend()
            self.temp_axes[2].grid(True)

            # Fitbereich markieren
            for ax in self.temp_axes[1:]:
                ax.axvspan(fit_min / 1000, fit_max / 1000, color="gray", alpha=0.14)

            self.temp_fig.tight_layout()
            self.temp_canvas.draw()
            self.temp_status.config(text="Fit erfolgreich durchgeführt!", foreground="green")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.temp_status.config(text="Fehler beim Fit!", foreground="red")

    def setup_radiosonde_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Radiosonde-Download")

        input_frame = ttk.LabelFrame(frame, text="Einstellungen")
        input_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(input_frame, text="Station ID:").grid(row=0, column=0)
        self.rs_station = ttk.Entry(input_frame, width=10)
        self.rs_station.insert(0, "11120")
        self.rs_station.grid(row=0, column=1)

        ttk.Label(input_frame, text="Datum [YYYYMMDD]:").grid(row=0, column=2)
        self.rs_date = ttk.Entry(input_frame, width=10)
        self.rs_date.insert(0, "20240901")
        self.rs_date.grid(row=0, column=3)

        ttk.Label(input_frame, text="UTC Stunde (00/12):").grid(row=0, column=4)
        self.rs_hour = ttk.Entry(input_frame, width=5)
        self.rs_hour.insert(0, "00")
        self.rs_hour.grid(row=0, column=5)

        # Speicherpfad
        ttk.Label(input_frame, text="Speicherpfad:").grid(row=1, column=0, sticky="w")
        self.rs_savepath = ttk.Entry(input_frame, width=50)
        path_default = os.path.join(BASE_DIR, "radiosonde")
        self.rs_savepath.insert(0, path_default)
        self.rs_savepath.grid(row=1, column=1, columnspan=4, sticky="we", padx=5)

        ttk.Button(input_frame, text="Ordner wählen", command=self.choose_rs_dir).grid(row=1, column=5, padx=5)

        ttk.Button(frame, text="Download starten", command=self.download_rs).pack(pady=5)

        # Kopierbares Feld für Ergebnis-Dateipfad
        result_frame = ttk.Frame(frame)
        result_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(result_frame, text="Gespeicherte Datei:").pack(side="left")
        self.rs_savedfile = ttk.Entry(result_frame, width=80, state="readonly")
        self.rs_savedfile.pack(side="left", padx=5, fill="x", expand=True)
        self.rs_status = ttk.Label(frame, text="Bereit.", foreground="blue")
        self.rs_status.pack()

    def choose_rs_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.rs_savepath.delete(0, tk.END)
            self.rs_savepath.insert(0, path)

    def download_rs(self):
        try:
            station = self.rs_station.get().strip()
            date = self.rs_date.get().strip()
            hour = self.rs_hour.get().strip()
            save_dir = self.rs_savepath.get().strip()

            if not os.path.isdir(save_dir):
                messagebox.showerror("Fehler", f"Ungültiger Speicherpfad:\n{save_dir}")
                return

            df, datetime_str = download_radiosonde_csv(station, date, hour)
            if df is None:
                self.rs_status.config(text="Keine Daten verfügbar!", foreground="red")
                return

            filename = f"sounding_{station}_{date}_{hour}UTC.csv"
            full_path = os.path.join(save_dir, filename)

            # Ordner anlegen, falls er nicht existiert
            os.makedirs(save_dir, exist_ok=True)

            df.to_csv(full_path, index=False)


            # Ergebnisfeld füllen 
            self.rs_savedfile.config(state="normal")
            self.rs_savedfile.delete(0, tk.END)
            self.rs_savedfile.insert(0, full_path)
            self.rs_savedfile.config(state="readonly")

            self.rs_status.config(
                text=f"Daten gespeichert: radiosonde/{filename}", foreground="green"
            )
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.rs_status.config(text="Fehler beim Download!", foreground="red")

    # sauberes Beenden
    def on_closing(self):
        try:
            self.wv_canvas.get_tk_widget().destroy()
            self.temp_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        plt.close("all")
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
