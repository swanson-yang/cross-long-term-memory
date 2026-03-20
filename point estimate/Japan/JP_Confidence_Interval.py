import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fractal_analysis import estimator
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator

import os
os.chdir(r"XXX\Longterm memory")  # <- change if needed

from ractional_Brownian_Field_main import RandomFieldSimulator
from scipy.special import gamma

def C_H(H: float) -> float:
    if np.abs(H - 0.5) < 0.0001:
        return float(np.pi)
    else:
        return float(gamma(2 - 2 * H) * np.cos(np.pi * H) / H / (1 - 2 * H))


def fbm_rf_cov(t_0: float, H_1: float, H_2: float) -> float:
    t_res = t_0 ** (H_1 + H_2) + t_0 ** (H_1 + H_2)
    return float(t_res * C_H((H_1 + H_2) / 2.0) / np.sqrt(C_H(H_1) * C_H(H_2)) / 2.0)

def load_series_from_excel(
    path: str,
    date_col_guess=("date", "Date", "DATE"),
    value_col: str | None = None,
) -> pd.DataFrame:

    df = pd.read_excel(path)

    # Find date column
    date_col = None
    for c in df.columns:
        if str(c) in date_col_guess or "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Cannot find a date column in: {path}. Columns={df.columns.tolist()}")

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Find value column
    if value_col is None:
        candidates = [c for c in df.columns if c != "date"]
        log_candidates = [c for c in candidates if "log" in str(c).lower()]
        if log_candidates:
            value_col = log_candidates[0]
        else:
            raise ValueError("No log column found in dataset.")

    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' not found in {path}. Columns={df.columns.tolist()}")

    out = df[["date", value_col]].rename(columns={value_col: "value"}).copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return out


def merge_three_series_on_date(sp: pd.DataFrame, ir: pd.DataFrame, vx: pd.DataFrame) -> pd.DataFrame:
    df = sp.rename(columns={"value": "log_price"}).merge(
        ir.rename(columns={"value": "log_interest"}), on="date", how="inner"
    ).merge(
        vx.rename(columns={"value": "log_vol"}), on="date", how="inner"
    )
    return df.sort_values("date").reset_index(drop=True)


def estimate_h_qv(series: pd.Series, alpha: float = 0.2) -> float:

    x = pd.to_numeric(series, errors="coerce").dropna().values
    if len(x) < 50:
        raise ValueError(f"Series too short for QV estimator: n={len(x)}")

    est = QvHurstEstimator(mbm_series=x, alpha=alpha)
    return float(np.nanmean(est.holder_exponents))


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

    rng = np.random.default_rng(seed)

    d = len(H_vec)
    if d < 2:
        raise ValueError("Need at least 2 H values to simulate a field cross-section.")

    t0 = 1.0 / float(n)

    # Cross-cov matrix across H values at time t0
    Sigma = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = fbm_rf_cov(t0, float(H_vec[i]), float(H_vec[j]))

    # numerical symmetry safeguard
    Sigma = (Sigma + Sigma.T) / 2.0

    paths = np.empty((B, d, n), dtype=float)

    for b in range(B):
        init_vals = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma)

        for i in range(d):
            sim = RandomFieldSimulator(
                sample_size=n,
                hurst_parameter=float(H_vec[i]),
                initial_value=float(init_vals[i]),
                tmax=float(tmax),
                FBM_cov_md=int(FBM_cov_md),
                rf_factor=float(rf_factor),
            )
            x = np.asarray(sim.get_self_similar_process(), dtype=float).reshape(-1)
            if len(x) != n:
                raise RuntimeError(f"Simulator returned length {len(x)} != n={n}")
            paths[b, i, :] = x

    return paths


def bootstrap_ci_by_period(
    df: pd.DataFrame,
    periods: list[tuple[str, str, str]],
    *,
    alpha_qv: float = 0.2,
    B: int = 300,
    ci_level: float = 0.90,
    FBM_cov_md: int = 1,   # Fractional Brownian field
    rf_factor: float = 0.7,
    seed: int = 1234,
) -> pd.DataFrame:

    out_rows = []
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q

    for label, start, end in periods:
        dfi = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
        dfi = dfi.dropna(subset=["log_price", "log_interest", "log_vol"]).reset_index(drop=True)

        n = len(dfi)
        if n < 80:
            raise ValueError(f"Period '{label}' too short after merge: n={n}")

        # Estimate H hats
        H1_hat = estimate_h_qv(dfi["log_price"], alpha=alpha_qv)
        H2_hat = estimate_h_qv(dfi["log_vol"], alpha=alpha_qv)
        H3_hat = estimate_h_qv(dfi["log_interest"], alpha=alpha_qv)

        H_vec = np.array([H1_hat, H2_hat, H3_hat], dtype=float)

        # Simulate surrogate fields
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

        # Percentile CIs
        for name, Hhat, col in [
            ("H1 (stock: log_price)", H1_hat, 0),
            ("H2 (vol: log_vol)", H2_hat, 1),
            ("H3 (rate: log_interest)", H3_hat, 2),
        ]:
            lo = float(np.quantile(H_boot[:, col], lo_q))
            hi = float(np.quantile(H_boot[:, col], hi_q))
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


if __name__ == "__main__":
    NIKKEI_XLSX = r"XXX/nikkei225_daily_close_201412_202512.xlsx"  # <- change if needed
    NKVI_XLSX   = r"XXX/Nikkei_Volatility_Historical_Data.xlsx"   # <- change if needed
    JGB_XLSX    = r"XXX/jgb_10y_yield_daily_201412_202512.xlsx"  # <- change if needed

    # Load as (date, value)
    px = load_series_from_excel(NIKKEI_XLSX, value_col="log_price")
    vx = load_series_from_excel(NKVI_XLSX,   value_col="log_vol")
    ir = load_series_from_excel(JGB_XLSX,    value_col="log_interest")

    # Merge on common dates
    df = merge_three_series_on_date(px, ir, vx)
    print("Merged rows =", len(df), "from", df["date"].min().date(), "to", df["date"].max().date())

    # sub-Periods
    periods = [
        ("Period 1", "2014-12-01", "2016-07-08"),
        ("Period 2", "2016-07-11", "2020-01-20"),
        ("Period 3", "2020-01-21", "2021-02-16"),
        ("Period 4", "2021-02-17", "2023-01-04"),
        ("Period 5", "2023-01-05", "2024-08-05"),
        ("Period 6", "2024-08-06", "2025-12-11"),
    ]

    ci_table = bootstrap_ci_by_period(
        df=df,
        periods=periods,
        alpha_qv=0.2,
        B=300,              
        ci_level=0.90,
        FBM_cov_md=1,       # Fractional Brownian field
        rf_factor=0.7,
        seed=1234,
    )

    print("\n=== CI TABLE (JAPAN) ===")
    print(ci_table)

    out_csv = "hurst_CI_field_bootstrap_JP_90.csv"
    ci_table.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")





