import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import math
import pandas as pd

# Geometrische Höhe aus geopotentieller Höhe
def geometric_altitude_from_geopotential(H_m, Re_m=6378000):
    return H_m / (1.0 - H_m / Re_m)

# Radiosonde CSV einlesen (Temperatur)
def read_radiosonde(file_path):
    df = pd.read_csv(file_path)

    h_geopot = pd.to_numeric(df["geopotential height_m"], errors='coerce').values
    temp_C = pd.to_numeric(df["temperature_C"], errors='coerce').values
    temp_K = temp_C + 273.15

    # geometrische Höhe berechnen
    h_geom = geometric_altitude_from_geopotential(h_geopot)

    # gültige Werte behalten
    valid = ~np.isnan(h_geom) & ~np.isnan(temp_K)
    h_geom = h_geom[valid]
    temp_K = temp_K[valid]

    # auf a.g.l. umrechnen (Startpunkt abziehen)
    if len(h_geom) > 0:
        h_agl = h_geom - h_geom[0]
    else:
        h_agl = h_geom

    return h_agl, temp_K


# Q(T) 
def forward_Q_from_T(T, a, b, c):
    expo = a + b / T + c / (T ** 2)
    expo = np.clip(expo, -700, 700)  # Overflow verhindern
    return np.exp(expo)


# Inverse T(Q)
def fit_func(Q, a, b, c):
    Q = np.asarray(Q, dtype=float)
    T = np.full_like(Q, np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        inner = (b / (2 * c))**2 - (a - np.log(Q)) / c
        mask = inner >= 0
        if not np.any(mask):
            return T

        sqrt_inner = np.sqrt(inner[mask])
        denom = ( -b / (2 * c)) + sqrt_inner
        T[mask] = np.where(denom != 0, 1.0 / denom, np.nan)

    return T


# Overlap-Auswertung
def eval_overlap(expr, x):
    allowed_names = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"x": x, "math": math, "np": np})
    return eval(expr, {"__builtins__": {}}, allowed_names)


# Haupt-Fit
def perform_fit(netcdf_file, sonde_file, fit_min, fit_max,
                plot_min=None, plot_max=None,
                params=None, overlap_expr=None):

    # Lidar-Daten
    ds = xr.open_dataset(netcdf_file, engine="netcdf4")
    rr1 = ds["RR1"].isel(time=0).values
    rr2 = ds["RR2"].isel(time=0).values

    # Range ist direkt Höhe über Grund (a.g.l.)
    heights_agl = ds["Range"].values
    q_ana = np.where(rr1 > 0, rr2 / rr1, np.nan)

    # Radiosonde CSV einlesen (jetzt a.g.l.)
    height_sonde_agl, temp_sonde = read_radiosonde(sonde_file)
    temp_interp = np.interp(heights_agl, height_sonde_agl, temp_sonde,
                            left=np.nan, right=np.nan)

    # Fitmaske
    fit_mask = ~np.isnan(q_ana) & ~np.isnan(temp_interp)
    fit_mask &= (heights_agl >= fit_min) & (heights_agl <= fit_max)
    if np.sum(fit_mask) == 0:
        raise ValueError("Keine gültigen Daten im Fitbereich!")

    Q_fit = q_ana[fit_mask]
    T_fit_data = temp_interp[fit_mask]
    H_fit = heights_agl[fit_mask]

    # Sortieren nach Temperatur
    sort_idx = np.argsort(T_fit_data)
    T_fit_data, Q_fit, H_fit = T_fit_data[sort_idx], Q_fit[sort_idx], H_fit[sort_idx]

    # Fit-Parameter
    if params:
        a, b, c = params
    else:
        try:
            Y = np.log(Q_fit)
            X = np.vstack([np.ones_like(T_fit_data),
                           1.0 / T_fit_data,
                           1.0 / (T_fit_data ** 2)]).T
            sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
            a0, b0, c0 = sol
        except Exception:
            a0, b0, c0 = 0.0, 1.0, 1.0

        p0 = [a0, b0, c0]
        T_grid = np.linspace(150, 320, 200)
        Q_grid = forward_Q_from_T(T_grid, *p0)
        if not np.all(np.isfinite(Q_grid)):
            p0 = [0.0, 1.0, 1.0]

        popt, _ = curve_fit(forward_Q_from_T, T_fit_data, Q_fit, p0=p0, maxfev=20000)
        a, b, c = popt

    # Overlap-Korrektur
    corr = eval_overlap(overlap_expr, heights_agl) if overlap_expr else np.zeros_like(heights_agl)
    q_corr = q_ana + corr

    # korrigierte Temperatur
    T_from_data_corr = fit_func(q_ana, a, b, c) + corr

    # Modellkurve im Fitbereich
    Q_model = forward_Q_from_T(T_fit_data, a, b, c)
    Q_model_corr = Q_model - np.interp(H_fit, heights_agl, corr)
    T_fit_curve = fit_func(Q_model_corr, a, b, c)

    # Residuen
    residuals = T_fit_data - T_fit_curve

    return (heights_agl, q_ana, temp_interp,
            H_fit, Q_fit, T_fit_curve,
            residuals, (a, b, c),
            T_from_data_corr)
