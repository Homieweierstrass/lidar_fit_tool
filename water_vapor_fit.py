import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import math
import pandas as pd

# Geometrische Höhe aus geopotentieller Höhe
def geometric_altitude_from_geopotential(H_m, Re_m=6378000):
    return H_m / (1.0 - H_m / Re_m)

# Radiosonde einlesen
def read_radiosonde(file_path):
    df = pd.read_csv(file_path)

    # Spalten in float, leere Strings → NaN
    h_geopot = pd.to_numeric(df["geopotential height_m"], errors='coerce').values
    mr = pd.to_numeric(df["mixing ratio_g/kg"], errors='coerce').values

    # Geometrische Höhe berechnen
    h_geom = geometric_altitude_from_geopotential(h_geopot)

    # Nur gültige Werte behalten
    valid = ~np.isnan(h_geom) & ~np.isnan(mr)
    h_geom = h_geom[valid]
    mr = mr[valid]

    # Höhe auf a.g.l. umrechnen: ersten Wert als Bodenhöhe abziehen
    if len(h_geom) > 0:
        h_agl = h_geom - h_geom[0]
    else:
        h_agl = h_geom

    return h_agl, mr

# Fitfunktion für Wasserdampf
def fit_func(Q, A):
    return A * Q

# Overlap-Korrektur auswerten
def eval_overlap(expr, x):
    x = np.array(x, dtype=np.float64)
    allowed_names = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"x": x, "math": math, "np": np})
    return eval(expr, {"__builtins__": {}}, allowed_names)

# Fit-Funktion
def perform_fit(netcdf_file, sonde_file, fit_min, fit_max, fit_a=None, overlap_expr=None):
    # Lidar-Daten einlesen
    ds = xr.open_dataset(netcdf_file, engine="netcdf4")
    rr1 = ds["RR1"].isel(time=0).values
    wv = ds["WV"].isel(time=0).values

    # Range ist bereits Höhe über Grund (a.g.l.)
    heights_agl = ds["Range"].values  

    # Mixing Ratio berechnen
    q_ana = np.where(rr1 > 0, wv / rr1, np.nan)

    # Radiosonde einlesen (jetzt a.g.l.)
    height_sonde_agl, mr_sonde = read_radiosonde(sonde_file)

    # Interpolation auf Lidar-Höhen (a.g.l.)
    mr_interp = np.interp(heights_agl, height_sonde_agl, mr_sonde, left=np.nan, right=np.nan)

    # Fitmaske
    fit_mask = ~np.isnan(q_ana) & ~np.isnan(mr_interp)
    fit_mask &= (heights_agl >= fit_min) & (heights_agl <= fit_max)
    if np.sum(fit_mask) == 0:
        raise ValueError("Keine gültigen Daten im Fitbereich!")

    Q_fit = q_ana[fit_mask]        
    MR_fit_data = mr_interp[fit_mask]
    H_fit = heights_agl[fit_mask]

    # Nach Q_fit sortieren 
    sort_idx = np.argsort(Q_fit)
    Q_fit = Q_fit[sort_idx]
    MR_fit_data = MR_fit_data[sort_idx]
    H_fit = H_fit[sort_idx]

    # Fit durchführen
    if fit_a is not None:
        A = fit_a
        MR_fit_curve = fit_func(Q_fit, A)
    else:
        popt, _ = curve_fit(fit_func, Q_fit, MR_fit_data, p0=[1.0], maxfev=10000)
        A = popt[0]
        MR_fit_curve = fit_func(Q_fit, A)

    residuals = MR_fit_data - MR_fit_curve

    # Overlap-Korrektur
    if overlap_expr:
        corr = eval_overlap(overlap_expr, heights_agl)
        q_corr = q_ana + corr
        q_corr[q_corr <= 0] = np.nan
    else:
        corr = np.zeros_like(heights_agl)
        q_corr = q_ana.copy()

    return heights_agl, q_ana, q_corr, mr_interp, H_fit, MR_fit_curve, residuals, A, corr
