import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from fractal_analysis import estimator
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator


# Working directory
# ============================================================
# Remark:
#   This line is machine-specific. When running the script on another computer,
#   replace the path by the folder containing the financial datasets and the
#   simulation code.

import os
os.chdir(r"XXX\Longterm memory")  # <- change if needed


# Custom simulator and special functions
# ============================================================
# RandomFieldSimulator:
#   This class generates sample paths of the fractional Brownian field used in
#   the bootstrap step. It is the core simulation device that produces
#   surrogate paths having prescribed Hurst parameters.
#
# gamma:
#   The Gamma function enters the normalizing constant appearing in the
#   covariance kernel of the fractional Brownian field.

from ractional_Brownian_Field_main import RandomFieldSimulator
from scipy.special import gamma


# Covariance ingredients for the fractional Brownian field
# ============================================================
# Main idea:
#   In the field-based bootstrap, the three simulated components must not be
#   treated as independent processes. Instead, they should be coupled through
#   a covariance structure reflecting the dependence induced by the Hurst
#   parameters.
#
#   The two functions below supply the constants used in that covariance function:
#
#   1. C_H(H):
#        a normalizing factor depending on a single Hurst parameter H.
#
#   2. fbm_rf_cov(t_0, H_1, H_2):
#        the covariance function between two field components evaluated at the same
#        time scale t_0 but with different Hurst parameters H1 and H2.

def C_H(H: float) -> float:
    """
    Normalizing constant associated with Hurst parameter H.

    Parameters:
        H : Hurst parameter, typically in (0,1).

    Returns:
        Value of the normalizing constant entering the covariance kernel.

    Remarks:
        The special case H = 1/2 is treated separately in order to avoid numerical
        instability in the formula and to recover the Brownian-motion benchmark.
    """
    if np.abs(H - 0.5) < 0.0001:
        return float(np.pi)
    else:
        return float(gamma(2 - 2 * H) * np.cos(np.pi * H) / H / (1 - 2 * H))


def fbm_rf_cov(t_0: float, H_1: float, H_2: float) -> float:
    """
    Covariance function of the fractional Brownian field.

    Parameters:
        t_0 : Reference time scale used in the bootstrap simulation.
        H_1, H_2 : Two Hurst parameters defining the pair of field components.

    Returns: 
        Covariance value coupling the two simulated components.

    Interpretation:
        This quantity is used to build the covariance matrix Sigma for the vector
        of initial values in the bootstrap simulation. Through Sigma, the three
        simulated series inherit a nontrivial cross-sectional dependence structure.
    """
    t_res = t_0 ** (H_1 + H_2) + t_0 ** (H_1 + H_2)
    return float(t_res * C_H((H_1 + H_2) / 2.0) / np.sqrt(C_H(H_1) * C_H(H_2)) / 2.0)


# Data loading from Excel
# ============================================================
# Purpose:
#   Each financial series is stored in a separate Excel file. The function
#   below loads one file, identifies a date column and a value column,
#   standardizes their names, and returns a clean two-column DataFrame: (date, value)


def load_series_from_excel(
    path: str,
    date_col_guess=("date", "Date", "DATE"),
    value_col: str | None = None,
) -> pd.DataFrame:

    """
    Load a single time series from an Excel file and standardize the output.

    Parameters:
        path : File path to the Excel dataset.
        date_col_guess : Candidate names for the date column.
        value_col : Name of the value column. If None, the function attempts to infer it.

    Returns:
        DataFrame with columns:
            - date
            - value
    """
    df = pd.read_excel(path)

    # Find date column automatically.
    date_col = None
    for c in df.columns:
        if str(c) in date_col_guess or "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Cannot find a date column in: {path}. Columns={df.columns.tolist()}")

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Find the numeric value column.
    # Preference is given to a column whose name contains "log", since the
    # empirical analysis is performed on log-transformed quantities.
    if value_col is None:
        candidates = [c for c in df.columns if c != "date"]
        log_candidates = [c for c in candidates if "log" in str(c).lower()]
        if log_candidates:
            value_col = log_candidates[0]   #Assume that the dataset contains only one log column
        else:
            raise ValueError("No log column found in dataset.")

    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' not found in {path}. Columns={df.columns.tolist()}")

    # Keep only date and value, coerce values to numeric, remove missing rows,
    # and sort chronologically.
    out = df[["date", value_col]].rename(columns={value_col: "value"}).copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return out


# Merge the three financial series
# ============================================================
# Main idea:
#   The stock index, interest-rate series, and volatility index may not share
#   exactly the same calendar due to holidays or missing observations.
#   Therefore, the three datasets are merged on their common dates only.
#
# Output:
#   A single aligned DataFrame containing:
#       - log_sp500
#       - log_interest
#       - log_vix

def merge_three_series_on_date(sp: pd.DataFrame, ir: pd.DataFrame, vx: pd.DataFrame) -> pd.DataFrame:
    """
    Merge on date; keep only common dates.
    """
    df = sp.rename(columns={"value": "log_sp500"}).merge(
        ir.rename(columns={"value": "log_interest"}), on="date", how="inner"
    ).merge(
        vx.rename(columns={"value": "log_vix"}), on="date", how="inner"
    )
    return df.sort_values("date").reset_index(drop=True)


# Hurst estimation by quadratic variation
# ============================================================
# Purpose:
#   For each series, the Hurst index is estimated using the quadratic
#   variation (QV) estimator.
#
# Interpretation:
#   - H > 0.5  : long-term memory
#   - H = 0.5  : Brownian benchmark
#   - H < 0.5  : anti-persistence

def estimate_h_qv(series: pd.Series, alpha: float = 0.2) -> float:
    """
    Estimate the Hurst parameter of a single time series by the QV estimator.

    Parameters:
        series : Input series.
        alpha : Smoothing / tuning parameter used in the estimator.

    Returns:
        Estimated Hurst parameter.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().values
    if len(x) < 50:
        raise ValueError(f"Series too short for QV estimator: n={len(x)}")

    est = QvHurstEstimator(mbm_series=x, alpha=alpha)
    return float(np.nanmean(est.holder_exponents))


# Surrogate-path simulation for the field bootstrap
# ============================================================
# Main idea:
#   After estimating the empirical Hurst vector H = (H1, H2, H3), we generate
#   bootstrap surrogate data that mimic the regularity and cross-sectional
#   dependence of the original three-dimensional system.
#
# Procedure:
#   1. Build the covariance matrix Sigma using the field covariance function.
#   2. Draw correlated initial values from N(0, Sigma).
#   3. For each component i, simulate a self-similar process with Hurst index H_i.
#
# The initial values are jointly sampled so that the three simulated series are tied together 
# while each marginal still evolves according to its own Hurst parameter.

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
    Purpose:
    This function generates B bootstrap surrogate samples for a 
    multi-dimensional process using a fractional Brownian field (FBF)
    framework. This serves as the core step in the field-based bootstrap procedure
    used to construct confidence intervals for the Hurst parameters.

    Parameters:
        H_vec : Vector of estimated Hurst parameters: H_vec = (H_1, ..., H_d)
            where each H_i corresponds to one time series component (e.g., stock, volatility, interest rate).
        n : Sample size (length of each time series).
        B : Number of bootstrap replications (number of surrogate samples).
        tmax : Time horizon of the simulated process.
        FBM_cov_md : Covariance model indicator for the fractional Brownian field.
        rf_factor : Scaling factor for the random field component.
        seed : Random seed for reproducibility of simulations.

    Returns:
        paths : np.ndarray of shape (B, d, n)
        Simulated surrogate paths:
            - B: number of bootstrap samples
            - d: number of series (here d = 3)
            - n: time length of each path

        paths[b, i, t] represents:
            the value at time t of component i in bootstrap sample b.

    Notes:
        The simulation proceeds in two stages:
            1. Cross-sectional dependence:
                A covariance matrix Σ is constructed using fbm_rf_cov,
                encoding dependence between components with Hurst indices H_i.

            2. Marginal simulation:
                For each component i, a self-similar process is simulated 
                with Hurst parameter H_i, initialized using correlated values.

        This ensures that both marginal properties and cross-dependence
        structure are preserved in the bootstrap samples.
    """

    # Initialize a reproducible random number generator for bootstrap simulation
    rng = np.random.default_rng(seed)

    # Number of components (e.g., 3 for stock, volatility, rate)
    d = len(H_vec)
    if d < 2:
        raise ValueError("Need at least 2 H values to simulate a field cross-section.")

    # Small time scale used to construct covariance
    t0 = 1.0 / float(n)

    # Build the covariance matrix across H values at time t0
    Sigma = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = fbm_rf_cov(t0, float(H_vec[i]), float(H_vec[j]))

    # Numerical safeguard: enforce symmetry explicitly.
    Sigma = (Sigma + Sigma.T) / 2.0

    # Allocate output array
    paths = np.empty((B, d, n), dtype=float)

    # Bootstrap simulation loop
    for b in range(B):

        # Generate correlated initial values, which injects cross-sectional dependence into the field
        init_vals = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma)

        # Simulate each component separately
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
            x = np.asarray(sim.get_self_similar_process(), dtype=float).reshape(-1)

            # Ensure correct length
            if len(x) != n:
                raise RuntimeError(f"Simulator returned length {len(x)} != n={n}")
            paths[b, i, :] = x

    return paths


# Confidence-interval construction period by period
# ============================================================
# Main workflow:
#   For each sub-period in the sample, the procedure is:
#
#   Step 1. Restrict the merged dataset to the chosen date interval.
#   Step 2. Estimate the three Hurst parameters from the empirical data.
#   Step 3. Use these estimates to simulate B surrogate field samples.
#   Step 4. Re-estimate the Hurst parameters on each surrogate sample.
#   Step 5. Construct percentile bootstrap confidence intervals.
#
# Interpretation:
#   Point estimates alone are often not sufficient because many estimated sums
#   H_i + H_j lie close to the critical threshold 1. The bootstrap confidence
#   intervals quantify the sampling uncertainty around each estimated Hurst
#   parameter and, later, around each pairwise sum.

def bootstrap_ci_by_period(
    df: pd.DataFrame,
    periods: list[tuple[str, str, str]],
    *,
    alpha_qv: float = 0.2,
    B: int = 300,
    ci_level: float = 0.90,  # Confidence level; determines lower/upper quantiles
    FBM_cov_md: int = 1,   # Fractional Brownian field
    rf_factor: float = 0.7,
    seed: int = 1234,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for H1, H2, and H3 in each period.

    Returns:
        A long-format table containing, for each period and each series,
        the point estimate H_hat and the corresponding bootstrap confidence interval.
    """
    out_rows = []
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q

    for label, start, end in periods:
        # Restrict to the current period and remove any remaining missing values.
        dfi = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
        dfi = dfi.dropna(subset=["log_sp500", "log_interest", "log_vix"]).reset_index(drop=True)

        n = len(dfi)
        if n < 80:
            raise ValueError(f"Period '{label}' too short after merge: n={n}")

        # Estimate the three empirical Hurst parameters.
        H1_hat = estimate_h_qv(dfi["log_sp500"], alpha=alpha_qv)
        H2_hat = estimate_h_qv(dfi["log_vix"], alpha=alpha_qv)
        H3_hat = estimate_h_qv(dfi["log_interest"], alpha=alpha_qv)

        H_vec = np.array([H1_hat, H2_hat, H3_hat], dtype=float)

        # Simulate bootstrap surrogate fields using the estimated H vector.
        paths = simulate_surrogate_field_paths(
            H_vec=H_vec,
            n=n,
            B=B,
            FBM_cov_md=FBM_cov_md,
            rf_factor=rf_factor,
            seed=seed,
        )

        # Re-estimate H on each surrogate series (marginal)
        H_boot = np.empty((B, 3), dtype=float)
        for b in range(B):
            H_boot[b, 0] = estimate_h_qv(pd.Series(paths[b, 0, :]), alpha=alpha_qv)
            H_boot[b, 1] = estimate_h_qv(pd.Series(paths[b, 1, :]), alpha=alpha_qv)
            H_boot[b, 2] = estimate_h_qv(pd.Series(paths[b, 2, :]), alpha=alpha_qv)

        # Construct percentile confidence intervals for each H_i.
        for name, Hhat, col in [
            ("H1 (stock: log_sp500)", H1_hat, 0),
            ("H2 (vol: log_vix)", H2_hat, 1),
            ("H3 (rate: log_interest)", H3_hat, 2),
        ]:
            lo = float(np.quantile(H_boot[:, col], lo_q))
            hi = float(np.quantile(H_boot[:, col], hi_q))
            """
            Alternative normal-approximation CI:
                se = float(np.std(H_boot[:, col], ddof=1))  # bootstrap SE
                z  = float(norm.ppf(hi_q))                  
                lo = float(Hhat - z * se)
                hi = float(Hhat + z * se)
            """
            out_rows.append(
                {
                    "period": label,
                    "start": start,
                    "end": end,
                    "series": name,
                    "n": n,
                    "H_hat": Hhat,
                    f"CI_{int(ci_level*100)}_lo": lo,
                    f"CI_{int(ci_level*100)}_hi": hi,
                    "B": B,
                    "alpha_qv": alpha_qv,
                    "FBM_cov_md": FBM_cov_md,
                    "rf_factor": rf_factor,
                }
            )

    return pd.DataFrame(out_rows)


# Main execution part
# ============================================================
# Steps:
#   1. Read the three U.S. datasets from Excel files.
#   2. Merge them on common dates.
#   3. Define the economically motivated sub-periods.
#   4. Run the field-based bootstrap CI procedure.
#   5. Print the resulting table and save it as a CSV file.
#
# Output:
#   A CSV file containing Hurst estimates and confidence intervals for:
#       - stock index
#       - volatility index
#       - interest rate
#
# This CSV file is then used by the companion script US_CI_Matrix.py to build
# pairwise confidence-interval matrices for H_i + H_j.

if __name__ == "__main__":
    SP500_XLSX = r"XXX/SP500_daily_close_2014-12_to_2025-12_STOOQ.xlsx"  # <- change if needed
    VIX_XLSX   = r"XXX/VIX_daily_close_2014-12_to_2025-12_FRED.xlsx"    # <- change if needed
    IR_XLSX    = r"XXX/US_10Y_DGS10_daily_2014-12_to_2025-12.xlsx"    # <- change if needed

    # Load each dataset as a standardized (date, value) table.
    sp = load_series_from_excel(SP500_XLSX, value_col="log_sp500")
    vx = load_series_from_excel(VIX_XLSX,   value_col="log_vix")
    ir = load_series_from_excel(IR_XLSX,    value_col="log_interest")

    # Merge the three series and report the date coverage after synchronization.
    df = merge_three_series_on_date(sp, ir, vx)
    print("Merged rows =", len(df), "from", df["date"].min().date(), "to", df["date"].max().date())

    # Period partition used in the U.S. empirical study.
    periods = [
        ("Period 1", "2014-12-01", "2016-02-11"),
        ("Period 2", "2016-02-12", "2020-02-19"),
        ("Period 3", "2020-02-20", "2021-12-29"),
        ("Period 4", "2021-12-30", "2022-10-12"),
        ("Period 5", "2022-10-13", "2025-12-31"),
    ]

    # Run bootstrap confidence-interval estimation.
    ci_table = bootstrap_ci_by_period(
        df=df,
        periods=periods,
        alpha_qv=0.2,
        B=300,              
        ci_level=0.90,
        FBM_cov_md=1,       
        rf_factor=0.7,      
        seed=1234,
    )

    print("\n=== CI TABLE ===")
    print(ci_table)

    out_csv = "hurst_CI_field_bootstrap_US_90.csv"
    ci_table.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")






