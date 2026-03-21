"""
Rolling-window field-bootstrap inference for time-varying Hurst paths
======================================================================

Purpose:
This script estimates time-varying Hurst parameters for three U.S. financial
series:

    1. stock prices      : log_sp500
    2. market volatility : log_vix
    3. interest rates    : log_interest

using a rolling-window quadratic-variation (QV) estimator. Within each rolling
window, the script then applies a field-based bootstrap procedure to construct
confidence intervals for:

    - each marginal Hurst parameter H_1, H_2, H_3,
    - each pairwise sum H_i + H_j.

The script therefore provides a time-varying view of both marginal persistence and cross persistence.

Main workflow:
    1. Load and clean three daily financial time series from Excel files.
    2. Merge the series on common calendar dates.
    3. Slide a rolling window across the merged sample.
    4. In each window:
       - estimate H_1, H_2, H_3 by the QV estimator,
       - simulate surrogate fractional Brownian fields,
       - re-estimate H on the bootstrap surrogates,
       - build confidence intervals for H_i and H_i + H_j.
    5. Save the rolling estimates to CSV.
    6. Plot confidence bands for all pairwise Hurst sums.

Outputs:
    - CSV file containing rolling point estimates and CI bands.
    - One 6-panel figure for all pairwise sums.
    - Six separate figures, one for each pairwise sum.

Notes:
    - This implementation assumes exactly three series throughout.
    - The random field simulator is imported from a local Python file.
"""

# A) Imports
# ======================================================================

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator



# B) Uniquely defined file paths and global configuration
# ======================================================================

# Paths of the three Excel files used in the U.S. application.
SP500_XLSX = r"XXX/SP500_daily_close_2014-12_to_2025-12_STOOQ.xlsx"     # <- change if needed
VIX_XLSX   = r"XXX/VIX_daily_close_2014-12_to_2025-12_FRED.xlsx"     # <- change if needed
IR_XLSX    = r"XXX/US_10Y_DGS10_daily_2014-12_to_2025-12.xlsx"     # <- change if needed

# Optional explicit date-column name.
# If COL_DATE is None, the code auto-detects the date column.
COL_DATE  = None  

# Explicit names of the value columns to be extracted from each file.
# These are the log-transformed series used in the empirical analysis.
COL_PRICE = "log_sp500"
COL_VOL   = "log_vix"
COL_RATE  = "log_interest"

# Rolling-window settings
# Length of each rolling window.
WINDOW = 252     # Use a rolling window of 252 observations (1 trading year) to estimate Hurst parameters

# Number of observations by which the window is shifted each time.
STRIDE = 5          # Updating the window every 5 observations (weekly) to capture time-varying dependence.
B = 100             # Number of bootstrap replications per rolling window.
CI_LEVEL = 0.90     # Confidence level for all bootstrap confidence intervals.
ALPHA_QV = 0.2      # Tuning parameter for the QV Hurst estimator.

# Field simulation settings
# In the RandomFieldSimulator, 1 indicates the fractional Brownian field setting.
FBM_cov_md = 1      # 1 = fractional Brownian field
RF_FACTOR = 0.7     # Strength/scaling parameter for the random-field component.
SIM_SEED = 1234     # Global simulation seed used to make the rolling bootstrap reproducible.

# If True, use percentile bootstrap intervals.
# If False, use a normal/Wald approximation with bootstrap standard errors.
USE_PERCENTILE = True

# US subperiods
# This section should be updated to reflect the final plotted time-varying confidence interval bands.
PERIODS = [
    ("Period 1", "2015-12-05", "2017-08-01"),       # <- change if needed
    ("Period 2", "2017-08-02", "2018-09-01"),       # <- change if needed
    ("Period 3", "2018-09-02", "2020-02-01"),       # <- change if needed
    ("Period 4", "2020-02-02", "2023-09-01"),       # <- change if needed
    ("Period 5", "2023-09-02", "2025-12-31"),       # <- change if needed
]


# C) Robust import of the RandomFieldSimulator
# ======================================================================

# Absolute path to the local module that defines RandomFieldSimulator.
RF_PATH = r"XXX/ractional_Brownian_Field_main.py"  # <- change if needed

def import_random_field_simulator():
    """
    Import the RandomFieldSimulator class either from a direct module import
    or from an explicit file path.

    Purpose:
        This helper function makes the script robust to different local setups.
        If the simulator file is not installed as a package, it can still be
        imported from its exact file location.

    Returns:
        RandomFieldSimulator : The simulator class used to generate self-similar surrogate paths.

    Notes:
        - If RF_PATH is None, the function falls back to a standard import.
        - Otherwise, it performs an explicit import-by-path.
    """
    if RF_PATH is None:
        from ractional_Brownian_Field_main import RandomFieldSimulator
        return RandomFieldSimulator

    # Dynamically load the fractional Brownian field simulator from a given file path.
    # This allows importing RandomFieldSimulator even if the file is not in the Python path.
    spec = importlib.util.spec_from_file_location("rf_module", RF_PATH)

    # Create a module object from the specification
    rf_module = importlib.util.module_from_spec(spec)

    # Execute the module (runs the code inside the file)
    spec.loader.exec_module(rf_module)

    # Return the simulator class used for generating surrogate sample paths
    return rf_module.RandomFieldSimulator

# Load the simulator once so it can be used throughout the script.
RandomFieldSimulator = import_random_field_simulator()



# D) Data cleaning and preprocessing
# ======================================================================

def _detect_date_col(df: pd.DataFrame) -> str:
    """
    Detect the date column in a raw Excel DataFrame.

    Parameters:
        df: Raw input table loaded from Excel.

    Returns:
        str: Name of the detected date column.

    Notes:
        - The function first searches for any column whose name contains
          the substring "date" (case-insensitive).
        - If no such column is found, it falls back to the first column.
    """
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    # first column
    return df.columns[0]

def _detect_value_col(df: pd.DataFrame, preferred: list[str]) -> str:
    """
    Detect the numeric value column in the Excel DataFrame.

    Parameters:
        df : Raw input table loaded from Excel.
        preferred : Ordered list of preferred candidate names for the value column.
        The function first tries exact matches, then case-insensitive matches,
        and finally falls back to the column with the largest number of valid
        numeric observations.

    Returns:
        str: Name of the selected value column.
    """

    # Exact match has highest priority.
    for name in preferred:
        if name is not None and name in df.columns:
            return name

    # case-insensitive match
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name is None:
            continue
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # If all preferred names fail, choose the most numeric column
    # among the non-date columns.
    date_col = _detect_date_col(df)
    best_col = None
    best_cnt = -1
    for c in df.columns:
        if c == date_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        cnt = int(s.notna().sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_col = c
    if best_col is None:
        raise ValueError("Cannot auto-detect a numeric value column.")
    return best_col

def load_series_from_excel(path: str, value_col: str | None, preferred_names: list[str]) -> pd.DataFrame:
    """
    Load one time series from an Excel file and standardize its format.

    Parameters:
        path : Path to the Excel file containing one financial time series.
        value_col : Explicit name of the value column to use.
                    If None or invalid, the function attempts automatic detection.
        preferred_names : Candidate column names, ordered by preference, used during automatic
                          value-column detection.

    Returns:
    pd.DataFrame: A cleaned DataFrame with exactly two columns:
                    - date  : datetime64
                    - value : numeric

    Notes:
        The function:
        - detects date and value columns,
        - renames them to a common format,
        - converts dates and values to proper types,
        - drops missing observations,
        - sorts rows chronologically.
    """
    df = pd.read_excel(path)

    dcol = COL_DATE if COL_DATE in (df.columns if COL_DATE is not None else []) else _detect_date_col(df)
    vcol = value_col if (value_col is not None and value_col in df.columns) else _detect_value_col(df, preferred_names)

    out = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return out

def merge_three_series_on_date(price: pd.DataFrame, rate: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three cleaned series on their common calendar dates.

    Parameters:
        price : Cleaned stock-price DataFrame with columns (date, value).
        rate : Cleaned interest-rate DataFrame with columns (date, value).
        vol : Cleaned volatility DataFrame with columns (date, value).

    Returns:
        A merged DataFrame with columns:
            - date
            - log_sp500
            - log_interest
            - log_vix

    Notes:
        The merge uses an inner join on date, so only dates observed in all three
        series are retained. This is essential because all rolling-window and
        pairwise calculations require aligned observations.
    """
    df = price.rename(columns={"value": "log_sp500"}).merge(
        rate.rename(columns={"value": "log_interest"}), on="date", how="inner"
    ).merge(
        vol.rename(columns={"value": "log_vix"}), on="date", how="inner"
    )
    return df.sort_values("date").reset_index(drop=True)


# E) Hurst estimation and field covariance ingredients
# ======================================================================

def estimate_h_qv(series, alpha: float = 0.2) -> float:
    """
    Estimate the Hurst parameter of a one-dimensional series using the
    quadratic-variation (QV) estimator.

    Parameters:
        series : Input sample path for which the Hurst parameter is estimated.
        alpha : Default=0.2

    Returns:
        Estimated Hurst parameter, computed here as the average of the local
        Holder-exponent estimates returned by QvHurstEstimator.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        raise ValueError(f"Series too short for QV estimator: n={len(x)}")
    est = QvHurstEstimator(mbm_series=x, alpha=alpha)
    return float(np.nanmean(est.holder_exponents))


def C_H(H: float) -> float:
    """
    Compute the normalization constant C(H) appearing in the covariance
    function of the fractional Brownian field.
    """
    if np.abs(H - 0.5) < 0.0001:
        return float(np.pi)
    else:
        return float(gamma(2 - 2 * H) * np.cos(np.pi * H) / H / (1 - 2 * H))


def fbm_rf_cov(t_0: float, H_1: float, H_2: float) -> float:
    """
    Compute the covariance funtion of the fractional Brownian field for
    two components with Hurst parameters H_1 and H_2 at time scale t_0.

    Notes:
        This quantity is used to build the covariance matrix Sigma that injects
        cross-sectional dependence into the bootstrap surrogate field.
    """
    t_res = t_0 ** (H_1 + H_2) + t_0 ** (H_1 + H_2)
    return float(t_res * C_H((H_1 + H_2) / 2.0) / np.sqrt(C_H(H_1) * C_H(H_2)) / 2.0)


# F) Surrogate field simulation for the bootstrap
# ======================================================================

def simulate_surrogate_field_paths(
    H_vec: np.ndarray,
    n: int,
    B: int,
    *,
    tmax: float = 1.0,
    FBM_cov_md: int = 1,
    rf_factor: float = 0.7,
    seed: int | None = 1234,
) -> np.ndarray:
    """
    Simulate bootstrap surrogate paths under a fractional Brownian field model.

    Parameters:
        H_vec : Vector of point estimates of the Hurst parameters in the current rolling window.
                H_vec = [H_stock, H_vol, H_rate].
        n : Length of each simulated path, typically equal to the rolling window size.
        B : Number of bootstrap replications to generate.
        tmax : Time horizon of each simulated self-similar path.
        FBM_cov_md : 1 = fractional Brownian field
        rf_factor : Scaling parameter controlling the random-field component.
        seed : Seed for the NumPy random generator used in simulation.

    Returns:
        Array of shape (B, d, n), where:
            - B is the number of bootstrap replications,
            - d is the number of series (here d = 3),
            - n is the path length.

        paths[b, i, t] is the value at time index t of component i in the
        b-th bootstrap surrogate sample.
    """

    # Initialize a random number generator with a fixed seed to ensure reproducibility.
    rng = np.random.default_rng(seed)

    # Number of components in the field.
    d = len(H_vec)
    if d != 3:
        raise ValueError("This script assumes exactly 3 series (H1,H2,H3).")

    # Small reference time scale used in the covariance function formula.
    t0 = 1.0 / float(n)

    # Build the 3 x 3 cross-covariance matrix Sigma.
    Sigma = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = float(fbm_rf_cov(t0, float(H_vec[i]), float(H_vec[j])))

    # Explicitly symmetrize to reduce numerical asymmetry.
    Sigma = (Sigma + Sigma.T) / 2.0


    # Project Sigma back to a numerically positive-semidefinite matrix.
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-10)   # small floor
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Sigma = (Sigma + Sigma.T) / 2.0        # re-symmetrize

    # Allocate output container for all simulated paths.
    paths = np.empty((B, d, n), dtype=float)

    # Bootstrap simulation loop.
    for b in range(B):
        # Draw correlated initial values so that the three components inherit
        # a cross-sectional dependence structure from Sigma.
        init_vals = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma)

        # Simulate one self-similar path for each component separately.
        for i in range(d):
            sim = RandomFieldSimulator(
                sample_size=n,
                hurst_parameter=float(H_vec[i]),
                initial_value=float(init_vals[i]),
                tmax=float(tmax),
                FBM_cov_md=int(FBM_cov_md),
                rf_factor=float(rf_factor),
            )

            # Generate self-similar process
            x = sim.get_self_similar_process()
            x = np.asarray(x, dtype=float).reshape(-1)

            # Ensure correct length
            if len(x) != n:
                raise RuntimeError(f"Simulator returned length {len(x)} != n={n}")
            paths[b, i, :] = x

    return paths


# G) Rolling-window bootstrap CI builder
# ====================================================================== 

def rolling_ci_path_field_bootstrap(
    df: pd.DataFrame,
    *,
    window: int,
    stride: int,
    B: int,
    ci_level: float,
    alpha_qv: float,
    FBM_cov_md: int,
    rf_factor: float,
    seed: int,
    use_percentile: bool = True,
) -> pd.DataFrame:
    """
    Construct rolling-window paths of Hurst estimates and confidence intervals
    using the field-based bootstrap.

    Parameters:
        df : Merged DataFrame containing the aligned columns:
                - date      - log_sp500      - log_vix     - log_interest
        window : Rolling-window length.
        stride : Step size between consecutive windows.
        B : Number of bootstrap replications within each window.
        ci_level : Confidence level for all intervals, e.g. 0.90 for 90% CIs.
        alpha_qv : Tuning parameter used in the QV Hurst estimator.
        FBM_cov_md : Covariance model indicator for the fractional Brownian field.
        rf_factor : Random-field scaling parameter.
        seed : Master random seed used to generate window-specific seeds.
        use_percentile : If True, compute percentile bootstrap intervals.
                         If False, compute normal/Wald intervals using bootstrap standard deviations.

    Returns:
        Rolling-output table where each row corresponds to one window endpoint.
        The DataFrame contains:
            - the window dates,
            - point estimates and CI bands for H_stock, H_vol, H_rate,
            - point estimates and CI bands for all pairwise sums:
                S_H11, S_H12, S_H13, S_H22, S_H23, S_H33.

    Notes:
        For each rolling window, it performs:
            1. point estimation of H_1, H_2, H_3,
            2. field-based bootstrap simulation,
            3. re-estimation on surrogate samples,
            4. CI construction for both marginal H's and pairwise sums.

        The pairwise sums are interpreted as diagnostics for cross long-term
        memory. In particular, when the confidence band for H_i + H_j lies fully
        above 1, that window provides evidence in favor of cross long-term memory
        for that pair.
    """

    # Ensure chronological order before constructing rolling windows.
    df = df.sort_values("date").reset_index(drop=True)

    # Extract the three aligned numeric series as NumPy arrays for speed.
    Xp = df["log_sp500"].to_numpy(dtype=float)
    Xv = df["log_vix"].to_numpy(dtype=float)
    Xr = df["log_interest"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()

    # Total number of aligned observations.
    nT = len(df)
    if nT < window + 10:
        raise ValueError(f"Too few rows: n={nT}, need at least window={window} + buffer.")

    # Lower and upper quantile levels corresponding to the requested CI level.
    # Example: ci_level = 0.90 gives lo_q = 0.05 and hi_q = 0.95.
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q

    # Initialize a master random number generator (RNG), used only to produce
    # independent seeds for each rolling window, ensuring reproducibility while
    # avoiding identical bootstrap samples across windows.
    rng = np.random.default_rng(seed)

    # Container for the rolling output rows.
    out_rows = []

    # Pair labels and their component indices.
    pairs = [
        ("H11", 0, 0),  # H1+H1
        ("H12", 0, 1),
        ("H13", 0, 2),
        ("H22", 1, 1),  # H2+H2
        ("H23", 1, 2),
        ("H33", 2, 2),  # H3+H3
    ]

    # Slide the rolling window across the full sample.
    for end_idx in range(window - 1, nT, stride):
        start_idx = end_idx - window + 1

        # Slice the current rolling window for each series.
        w_price = Xp[start_idx : end_idx + 1]
        w_vol   = Xv[start_idx : end_idx + 1]
        w_rate  = Xr[start_idx : end_idx + 1]

        # point estimation of the three Hurst parameters.
        H_price = estimate_h_qv(w_price, alpha=alpha_qv)   # H1
        H_vol   = estimate_h_qv(w_vol,   alpha=alpha_qv)   # H2
        H_rate  = estimate_h_qv(w_rate,  alpha=alpha_qv)   # H3
        H_vec = np.array([H_price, H_vol, H_rate], dtype=float)

        # Simulate B surrogate field paths of length = window.
        # A fresh seed is generated for each rolling window so that windows are
        # reproducible but not forced to reuse identical random draws.
        local_seed = int(rng.integers(0, 2**31 - 1))
        paths = simulate_surrogate_field_paths(
            H_vec=H_vec,
            n=window,
            B=B,
            FBM_cov_md=FBM_cov_md,
            rf_factor=rf_factor,
            seed=local_seed,
        )

        # Re-estimate H on each bootstrap surrogate sample.
        H_boot = np.empty((B, 3), dtype=float)
        for b in range(B):
            H_boot[b, 0] = estimate_h_qv(paths[b, 0, :], alpha=alpha_qv)  # H1*
            H_boot[b, 1] = estimate_h_qv(paths[b, 1, :], alpha=alpha_qv)  # H2*
            H_boot[b, 2] = estimate_h_qv(paths[b, 2, :], alpha=alpha_qv)  # H3*

        # Build confidence intervals for H1, H2, H3.
        if use_percentile:
            # Percentile bootstrap intervals based on empirical quantiles.
            lo_price = float(np.quantile(H_boot[:, 0], lo_q))
            hi_price = float(np.quantile(H_boot[:, 0], hi_q))
            lo_vol   = float(np.quantile(H_boot[:, 1], lo_q))
            hi_vol   = float(np.quantile(H_boot[:, 1], hi_q))
            lo_rate  = float(np.quantile(H_boot[:, 2], lo_q))
            hi_rate  = float(np.quantile(H_boot[:, 2], hi_q))
        else:
            # Normal/Wald approximation based on bootstrap standard errors.
            z = 1.959963984540054  
            se_price = float(np.std(H_boot[:, 0], ddof=1))
            se_vol   = float(np.std(H_boot[:, 1], ddof=1))
            se_rate  = float(np.std(H_boot[:, 2], ddof=1))
            lo_price, hi_price = float(H_price - z * se_price), float(H_price + z * se_price)
            lo_vol,   hi_vol   = float(H_vol   - z * se_vol),   float(H_vol   + z * se_vol)
            lo_rate,  hi_rate  = float(H_rate  - z * se_rate),  float(H_rate  + z * se_rate)

        # Build point estimates of all pairwise Hurst sums.
        # Point estimates of sums
        S_hat = {
            "H11": H_price + H_price,
            "H12": H_price + H_vol,
            "H13": H_price + H_rate,
            "H22": H_vol + H_vol,
            "H23": H_vol + H_rate,
            "H33": H_rate + H_rate,
        }

        # Bootstrap distributions of the sums H_i^* + H_j^*.
        S_boot = {key: (H_boot[:, i] + H_boot[:, j]) for key, i, j in pairs}

        # Build confidence intervals for all pairwise sums.
        if use_percentile:
            S_ci = {
                key: (
                    float(np.quantile(S_boot[key], lo_q)),
                    float(np.quantile(S_boot[key], hi_q)),
                )
                for key, _, _ in pairs
            }
        else:
            z = 1.959963984540054
            S_ci = {}
            for key, _, _ in pairs:
                se = float(np.std(S_boot[key], ddof=1))
                S_ci[key] = (float(S_hat[key] - z * se), float(S_hat[key] + z * se))

        # Store one output row for the current rolling window.
        row = {
            "date": pd.to_datetime(dates[end_idx]),
            "window_start": pd.to_datetime(dates[start_idx]),
            "window_end": pd.to_datetime(dates[end_idx]),
            "window": int(window),
            "stride": int(stride),
            "B": int(B),
            "ci_level": float(ci_level),
            "alpha_qv": float(alpha_qv),
            "FBM_cov_md": int(FBM_cov_md),
            "rf_factor": float(rf_factor),
            "band_type": "percentile" if use_percentile else "sd_wald",

            # Rolling point estimate and CI for stock Hurst parameter.
            "H_stock": float(H_price),
            "H_stock_lo": float(lo_price),
            "H_stock_hi": float(hi_price),

             # Rolling point estimate and CI for volatility Hurst parameter.
            "H_vol": float(H_vol),
            "H_vol_lo": float(lo_vol),
            "H_vol_hi": float(hi_vol),

            # Rolling point estimate and CI for interest-rate Hurst parameter.
            "H_rate": float(H_rate),
            "H_rate_lo": float(lo_rate),
            "H_rate_hi": float(hi_rate),
        }

        # Add point estimates and CI bounds for all pairwise sums.
        for key, _, _ in pairs:
            row[f"S_{key}"] = float(S_hat[key])
            row[f"S_{key}_lo"] = float(S_ci[key][0])
            row[f"S_{key}_hi"] = float(S_ci[key][1])

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)


# H) Plotting functions
# ======================================================================

def add_subperiod_separators(ax, periods, *, color="red", linestyle=":", linewidth=1.2, alpha=0.9):
    """
    Add vertical separator lines marking the boundaries between predefined
    subperiods.

    Parameters:
        ax : Axis on which the separators are drawn.
        periods : List of period tuples of the form (label, start, end).
        color : Color of the separator lines.
        linestyle : Line style of the separators.
        linewidth : Width of the separator lines.
        alpha : Transparency level of the separators.
    """
    boundaries = [pd.to_datetime(p[1]) for p in periods[1:]]
    for dt in boundaries:
        ax.axvline(dt, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=10)


def plot_ci_band(ci_df, mid, lo, hi, periods, title, out_png, *, y_label="H sum", legend_label: str | None = None,):
    """
    Plot one rolling estimate together with its confidence band.

    Parameters:
        ci_df : Rolling-output DataFrame produced by rolling_ci_path_field_bootstrap.
        mid : Column name of the point-estimate series to plot.
        lo : Column name of the lower confidence-bound series.
        hi : Column name of the upper confidence-bound series.
        periods : Period definitions used to draw separator lines.
        title : Figure title.
        out_png : Output PNG filename.
        y_label : Label on the vertical axis.
        legend_label : Legend label for the point-estimate curve.

    Notes:
        This function is used for the six separate pairwise-sum figures.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if legend_label is None:
        legend_label = title

    ax.plot(ci_df["date"], ci_df[mid], linewidth=1.5, label=legend_label)
    ax.fill_between(ci_df["date"], ci_df[lo], ci_df[hi], alpha=0.25, label="CI band")
    add_subperiod_separators(ax, periods)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_six_sum_panels(ci_df, periods, out_png, *, title_map=None, y_label="H sum"):
    """
    Plot all six pairwise Hurst sums and their CI bands in one 3x2 panel figure.

    Parameters:
        ci_df : Rolling-output DataFrame produced by rolling_ci_path_field_bootstrap.
        periods : Period definitions used to draw separator lines.
        out_png : Output PNG filename for the combined panel figure.
        title_map : Optional mapping from series keys to subplot titles.
        y_label : Vertical-axis label shared conceptually across subplots.

    Notes:
        The six plotted sums are: H1+H1, H1+H2, H1+H3, H2+H2, H2+H3, H3+H3.
    """
    if title_map is None:
        title_map = {}
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    panels = [
        ("S_H11", "S_H11_lo", "S_H11_hi", title_map.get("S_H11", "H(Stock)+H(Stock)")),
        ("S_H12", "S_H12_lo", "S_H12_hi", title_map.get("S_H12", "H(Stock)+H(Volatility)")),
        ("S_H13", "S_H13_lo", "S_H13_hi", title_map.get("S_H13", "H(Stock)+H(Interest Rate)")),
        ("S_H22", "S_H22_lo", "S_H22_hi", title_map.get("S_H22", "H(Volatility)+H(Volatility)")),
        ("S_H23", "S_H23_lo", "S_H23_hi", title_map.get("S_H23", "H(Volatility)+H(Interest Rate)")),
        ("S_H33", "S_H33_lo", "S_H33_hi", title_map.get("S_H33", "H(Interest Rate)+H(Interest Rate)")),
    ]

    for ax, (mid, lo, hi, ttl) in zip(axes, panels):
        ax.plot(ci_df["date"], ci_df[mid], linewidth=1.5)
        ax.fill_between(ci_df["date"], ci_df[lo], ci_df[hi], alpha=0.25)
        add_subperiod_separators(ax, periods)
        ax.set_title(ttl)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Date")
    axes[-2].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# I) Main execution block
# ======================================================================

if __name__ == "__main__":
    print("=== Loading US datasets ===")

    # Load the three source files and standardize them to (date, value) form.
    price = load_series_from_excel(
        SP500_XLSX,
        value_col=COL_PRICE,
        preferred_names=["log_sp500", "log_price", "close", "log_close"],
    )
    vol = load_series_from_excel(
        VIX_XLSX,
        value_col=COL_VOL,
        preferred_names=["log_vix", "log_vol", "vix", "vol"],
    )
    rate = load_series_from_excel(
        IR_XLSX,
        value_col=COL_RATE,
        preferred_names=["log_interest", "interest", "yield", "rate", "log_rate"],
    )

    # Merge the three series so that all subsequent analysis is performed on
    # common dates only.
    df = merge_three_series_on_date(price, rate, vol)
    print("Merged rows:", len(df))
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("Columns:", df.columns.tolist())

    print("\n=== Rolling FIELD-bootstrap H(t) + CI, then pairwise sums + CI ===")

    # Build the rolling point estimates and CI bands.
    ci_df = rolling_ci_path_field_bootstrap(
        df,
        window=WINDOW,
        stride=STRIDE,
        B=B,
        ci_level=CI_LEVEL,
        alpha_qv=ALPHA_QV,
        FBM_cov_md=FBM_cov_md,
        rf_factor=RF_FACTOR,
        seed=SIM_SEED,
        use_percentile=USE_PERCENTILE,
    )

    # Save the full rolling-output table for later use in tables/plots.
    out_csv = "us_H_paths_CI_field_bootstrap_with_sums_90.csv"
    ci_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(ci_df.head())

    # Descriptive titles (Stock / Volatility / Interest Rate)
    sum_title_map = {
        "S_H11": "H(Stock) + H(Stock)",
        "S_H12": "H(Stock) + H(Volatility)",
        "S_H13": "H(Stock) + H(Interest Rate)",
        "S_H22": "H(Volatility) + H(Volatility)",
        "S_H23": "H(Volatility) + H(Interest Rate)",
        "S_H33": "H(Interest Rate) + H(Interest Rate)",
    }

    print("\n=== Plotting: combined 6-panel sums figure ===")
    plot_six_sum_panels(
        ci_df=ci_df,
        periods=PERIODS,
        out_png="us_H_pairwise_sums_CI_ALL_90.png",
        title_map=sum_title_map,
        y_label="H sum",
    )

    print("\n=== Plotting: six separate sum figures ===")
    sum_panels = [
        ("S_H11", "S_H11_lo", "S_H11_hi", sum_title_map["S_H11"], "us_sum_Stock_Stock_CI_90.png"),
        ("S_H12", "S_H12_lo", "S_H12_hi", sum_title_map["S_H12"], "us_sum_Stock_Volatility_CI_90.png"),
        ("S_H13", "S_H13_lo", "S_H13_hi", sum_title_map["S_H13"], "us_sum_Stock_InterestRate_CI_90.png"),
        ("S_H22", "S_H22_lo", "S_H22_hi", sum_title_map["S_H22"], "us_sum_Volatility_Volatility_CI_90.png"),
        ("S_H23", "S_H23_lo", "S_H23_hi", sum_title_map["S_H23"], "us_sum_Volatility_InterestRate_CI_90.png"),
        ("S_H33", "S_H33_lo", "S_H33_hi", sum_title_map["S_H33"], "us_sum_InterestRate_InterestRate_CI_90.png"),
    ]

    for mid, lo, hi, ttl, fname in sum_panels:
        plot_ci_band(
            ci_df,
            mid, lo, hi,
            PERIODS,
            title=f"US: {ttl} with CI band",
            out_png=fname,
            y_label="H sum",
        )

    print("\nSaved figures:")
    print(" - us_H_pairwise_sums_CI_ALL_90.png")
    for _, _, _, _, fname in sum_panels:
        print(" -", fname)

    print("\nDone.")





