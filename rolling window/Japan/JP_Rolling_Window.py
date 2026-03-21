import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gamma
from fractal_analysis.estimator.hurst_estimator import QvHurstEstimator


# File paths
NIKKEI_XLSX = r"XXX/nikkei225_daily_close_201412_202512.xlsx"   # <- change if needed
NVIX_XLSX   = r"XXX/Nikkei_Volatility_Historical_Data.xlsx"    # <- change if needed
JGB_XLSX    = r"XXX/jgb_10y_yield_daily_201412_202512.xlsx"    # <- change if needed

# Column choices
COL_DATE  = None  
COL_PRICE = "log_price"
COL_VOL   = "log_vol"
COL_RATE  = "log_interest"

# Rolling / Bootstrap settings
WINDOW    = 252     # 1 trading year
STRIDE    = 5      
B         = 100     
CI_LEVEL  = 0.90
ALPHA_QV  = 0.2


FBM_cov_md = 1
RF_FACTOR  = 0.7      
SIM_SEED   = 1234
USE_PERCENTILE = True  # True: percentile bands; False: SD/Wald bands


# Sub-periods
PERIODS = [
    ("Period 1", "2015-12-01", "2016-12-01"),      # <- change if needed
    ("Period 2", "2016-12-02", "2017-12-01"),      # <- change if needed
    ("Period 3", "2017-12-02", "2021-03-01"),      # <- change if needed
    ("Period 4", "2021-03-02", "2024-08-01"),      # <- change if needed
    ("Period 5", "2024-08-02", "2025-12-11"),      # <- change if needed
]


# Import RandomFieldSimulator robustly
RF_PATH = r"XXX/ractional_Brownian_Field_main.py"   # <- change if needed

def import_random_field_simulator():
    if RF_PATH is None:
        from ractional_Brownian_Field_main import RandomFieldSimulator
        return RandomFieldSimulator

    spec = importlib.util.spec_from_file_location("rf_module", RF_PATH)
    rf_module = importlib.util.module_from_spec(spec)
    sys.modules["rf_module"] = rf_module
    assert spec.loader is not None
    spec.loader.exec_module(rf_module)
    return rf_module.RandomFieldSimulator

RandomFieldSimulator = import_random_field_simulator()


def C_H(H: float) -> float:
    if np.abs(H - 0.5) < 0.0001:
        return float(np.pi)
    else:
        return float(gamma(2 - 2 * H) * np.cos(np.pi * H) / H / (1 - 2 * H))


def fbm_rf_cov(t_0: float, H_1: float, H_2: float) -> float:
    t_res = t_0 ** (H_1 + H_2) + t_0 ** (H_1 + H_2)
    return float(t_res * C_H((H_1 + H_2) / 2.0) / np.sqrt(C_H(H_1) * C_H(H_2)) / 2.0)


# D) Excel reader
def _detect_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    return df.columns[0]

def _detect_value_col(df: pd.DataFrame, preferred: list[str]) -> str:
    for name in preferred:
        if name is not None and name in df.columns:
            return name

    lower_map = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name is None:
            continue
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    date_col = _detect_date_col(df)
    best_col, best_cnt = None, -1
    for c in df.columns:
        if c == date_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        cnt = int(s.notna().sum())
        if cnt > best_cnt:
            best_cnt, best_col = cnt, c
    if best_col is None:
        raise ValueError("Cannot auto-detect a numeric value column.")
    return best_col

def load_series_from_excel(path: str, value_col: str | None, preferred_names: list[str]) -> pd.DataFrame:
    df = pd.read_excel(path)

    dcol = COL_DATE if (COL_DATE is not None and COL_DATE in df.columns) else _detect_date_col(df)
    vcol = value_col if (value_col is not None and value_col in df.columns) else _detect_value_col(df, preferred_names)

    out = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return out

def merge_three_series_on_date(price: pd.DataFrame, rate: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    df = (
        price.rename(columns={"value": "log_price"})
        .merge(rate.rename(columns={"value": "log_interest"}), on="date", how="inner")
        .merge(vol.rename(columns={"value": "log_vol"}), on="date", how="inner")
    )
    return df.sort_values("date").reset_index(drop=True)



# QV estimator
def estimate_h_qv(series, alpha: float = 0.2) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        raise ValueError(f"Series too short for QV estimator: n={len(x)}")
    est = QvHurstEstimator(mbm_series=x, alpha=alpha)
    return float(np.nanmean(est.holder_exponents))



# Simulate surrogate fields paths (uses the RandomFieldSimulator)
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
    if d != 3:
        raise ValueError("This script expects exactly 3 series (price, vol, rate).")

    t0 = 1.0 / float(n)


    Sigma = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            Sigma[i, j] = fbm_rf_cov(t0, float(H_vec[i]), float(H_vec[j]))
    Sigma = (Sigma + Sigma.T) / 2.0


    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-10)   # small floor
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Sigma = (Sigma + Sigma.T) / 2.0        # re-symmetrize

    # Draw correlated initial values via multivariate normal
    paths = np.empty((B, d, n), dtype=float)

    for b in range(B):
        init_vals = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma)

        for i in range(d):
            sim = RandomFieldSimulator(
                sample_size=int(n),
                hurst_parameter=float(H_vec[i]),
                initial_value=float(init_vals[i]),
                tmax=float(tmax),
                FBM_cov_md=int(FBM_cov_md),
                rf_factor=float(rf_factor),
            )

            x = sim.get_self_similar_process()
            x = np.asarray(x, dtype=float).reshape(-1)
            if len(x) != n:
                raise RuntimeError(f"Simulator returned length {len(x)} != n={n}")
            paths[b, i, :] = x

    return paths


# Rolling CI-path builder
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

    df = df.sort_values("date").reset_index(drop=True)

    Xp = df["log_price"].to_numpy(dtype=float)
    Xv = df["log_vol"].to_numpy(dtype=float)
    Xr = df["log_interest"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()

    nT = len(df)
    if nT < window + 10:
        raise ValueError(f"Too few rows: n={nT}, need at least window={window} + buffer.")

    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q

    rng = np.random.default_rng(seed)
    out_rows: list[dict] = []

    pairs = [
        ("H11", 0, 0),
        ("H12", 0, 1),
        ("H13", 0, 2),
        ("H22", 1, 1),
        ("H23", 1, 2),
        ("H33", 2, 2),
    ]

    for end_idx in range(window - 1, nT, stride):
        start_idx = end_idx - window + 1

        w_price = Xp[start_idx : end_idx + 1]
        w_vol   = Xv[start_idx : end_idx + 1]
        w_rate  = Xr[start_idx : end_idx + 1]

        H_price = estimate_h_qv(w_price, alpha=alpha_qv)  # H1
        H_vol   = estimate_h_qv(w_vol,   alpha=alpha_qv)  # H2
        H_rate  = estimate_h_qv(w_rate,  alpha=alpha_qv)  # H3
        H_vec = np.array([H_price, H_vol, H_rate], dtype=float)

        local_seed = int(rng.integers(0, 2**31 - 1))
        paths = simulate_surrogate_field_paths(
            H_vec=H_vec,
            n=window,
            B=B,
            FBM_cov_md=FBM_cov_md,
            rf_factor=rf_factor,
            seed=local_seed,
        )

        H_boot = np.empty((B, 3), dtype=float)
        for b in range(B):
            H_boot[b, 0] = estimate_h_qv(paths[b, 0, :], alpha=alpha_qv)
            H_boot[b, 1] = estimate_h_qv(paths[b, 1, :], alpha=alpha_qv)
            H_boot[b, 2] = estimate_h_qv(paths[b, 2, :], alpha=alpha_qv)

        # compute CI for H1/H2/H3
        if use_percentile:
            lo_price = float(np.quantile(H_boot[:, 0], lo_q))
            hi_price = float(np.quantile(H_boot[:, 0], hi_q))
            lo_vol   = float(np.quantile(H_boot[:, 1], lo_q))
            hi_vol   = float(np.quantile(H_boot[:, 1], hi_q))
            lo_rate  = float(np.quantile(H_boot[:, 2], lo_q))
            hi_rate  = float(np.quantile(H_boot[:, 2], hi_q))
        else:
            # SD/Wald (normal approx)
            z = 1.959963984540054  # norm.ppf(0.975)
            se_price = float(np.std(H_boot[:, 0], ddof=1))
            se_vol   = float(np.std(H_boot[:, 1], ddof=1))
            se_rate  = float(np.std(H_boot[:, 2], ddof=1))
            lo_price, hi_price = float(H_price - z * se_price), float(H_price + z * se_price)
            lo_vol,   hi_vol   = float(H_vol   - z * se_vol),   float(H_vol   + z * se_vol)
            lo_rate,  hi_rate  = float(H_rate  - z * se_rate),  float(H_rate  + z * se_rate)

        # Sums point estimates
        S_hat = {
            "H11": H_price + H_price,
            "H12": H_price + H_vol,
            "H13": H_price + H_rate,
            "H22": H_vol + H_vol,
            "H23": H_vol + H_rate,
            "H33": H_rate + H_rate,
        }
        S_boot = {key: (H_boot[:, i] + H_boot[:, j]) for key, i, j in pairs}

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

            "H_price": float(H_price),
            "H_price_lo": float(lo_price),
            "H_price_hi": float(hi_price),

            "H_vol": float(H_vol),
            "H_vol_lo": float(lo_vol),
            "H_vol_hi": float(hi_vol),

            "H_interest": float(H_rate),
            "H_interest_lo": float(lo_rate),
            "H_interest_hi": float(hi_rate),

        }

        for key, _, _ in pairs:
            row[f"S_{key}"] = float(S_hat[key])
            row[f"S_{key}_lo"] = float(S_ci[key][0])
            row[f"S_{key}_hi"] = float(S_ci[key][1])

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)



def add_subperiod_separators(
    ax,
    periods,
    *,
    color="red",
    linestyle=":",
    linewidth=1.2,
    alpha=0.9
):

    boundaries = [pd.to_datetime(p[1]) for p in periods[1:]] 
    for dt in boundaries:
        ax.axvline(
            dt,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=10
        )


def plot_ci_band(
    ci_df: pd.DataFrame,
    H_col: str,
    lo_col: str,
    hi_col: str,
    periods,
    title: str,
    out_png: str,
    *,
    y_label: str = "Value",
    legend_label: str | None = None,
):

    fig, ax = plt.subplots(figsize=(12, 4))

    if legend_label is None:
        legend_label = title

    ax.plot(ci_df["date"], ci_df[H_col], linewidth=1.5, label=legend_label)
    ax.fill_between(ci_df["date"], ci_df[lo_col], ci_df[hi_col], alpha=0.25, label="CI band")

    add_subperiod_separators(ax, periods)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_six_sum_panels(
    ci_df: pd.DataFrame,
    periods,
    out_png: str,
    *,
    title_map: dict[str, str] | None = None,
    y_label: str = "H sum",
):

    if title_map is None:
        title_map = {}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    panels = [
        ("S_H11", "S_H11_lo", "S_H11_hi", title_map.get("S_H11", "H1 + H1")),
        ("S_H12", "S_H12_lo", "S_H12_hi", title_map.get("S_H12", "H1 + H2")),
        ("S_H13", "S_H13_lo", "S_H13_hi", title_map.get("S_H13", "H1 + H3")),
        ("S_H22", "S_H22_lo", "S_H22_hi", title_map.get("S_H22", "H2 + H2")),
        ("S_H23", "S_H23_lo", "S_H23_hi", title_map.get("S_H23", "H2 + H3")),
        ("S_H33", "S_H33_lo", "S_H33_hi", title_map.get("S_H33", "H3 + H3")),
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



# MAIN: run + save CSV + save 7 figures
if __name__ == "__main__":
    print("=== Loading Japan datasets ===")
    price = load_series_from_excel(
        NIKKEI_XLSX,
        value_col=COL_PRICE,
        preferred_names=["log_price", "logprice", "price", "close", "log_close"],
    )
    vol = load_series_from_excel(
        NVIX_XLSX,
        value_col=COL_VOL,
        preferred_names=["log_vol", "logvix", "vix", "vol", "nikkei_volatility"],
    )
    rate = load_series_from_excel(
        JGB_XLSX,
        value_col=COL_RATE,
        preferred_names=["log_interest", "interest", "yield", "rate", "log_rate"],
    )

    df = merge_three_series_on_date(price, rate, vol)
    print("Merged rows:", len(df))
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("Columns:", df.columns.tolist())

    print("\n=== Rolling FIELD-bootstrap CI paths for sums ===")
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

    out_csv = "japan_H_sums_paths_CI_field_bootstrap_90.csv"
    ci_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(ci_df.head())

    sum_title_map = {
        "S_H11": "H(Stock) + H(Stock)",
        "S_H12": "H(Stock) + H(Volatility)",
        "S_H13": "H(Stock) + H(Interest Rate)",
        "S_H22": "H(Volatility) + H(Volatility)",
        "S_H23": "H(Volatility) + H(Interest Rate)",
        "S_H33": "H(Interest Rate) + H(Interest Rate)",
    }


    # One combined figure with 6 panels

    print("\n=== Plotting: combined 6-panel sums figure (descriptive titles) ===")
    plot_six_sum_panels(
        ci_df=ci_df,
        periods=PERIODS,
        out_png="japan_H_pairwise_sums_CI_ALL_90.png",
        title_map=sum_title_map,   
        y_label="H sum",
    )


    # Six separate figures

    print("\n=== Plotting: six separate sum figures (descriptive titles) ===")
    sum_panels = [
        ("S_H11", "S_H11_lo", "S_H11_hi", sum_title_map["S_H11"], "japan_sum_Stock_Stock_CI_90.png"),
        ("S_H12", "S_H12_lo", "S_H12_hi", sum_title_map["S_H12"], "japan_sum_Stock_Volatility_CI_90.png"),
        ("S_H13", "S_H13_lo", "S_H13_hi", sum_title_map["S_H13"], "japan_sum_Stock_InterestRate_CI_90.png"),
        ("S_H22", "S_H22_lo", "S_H22_hi", sum_title_map["S_H22"], "japan_sum_Volatility_Volatility_CI_90.png"),
        ("S_H23", "S_H23_lo", "S_H23_hi", sum_title_map["S_H23"], "japan_sum_Volatility_InterestRate_CI_90.png"),
        ("S_H33", "S_H33_lo", "S_H33_hi", sum_title_map["S_H33"], "japan_sum_InterestRate_InterestRate_CI_90.png"),
    ]

    for mid, lo, hi, ttl, fname in sum_panels:
        plot_ci_band(
            ci_df=ci_df,
            H_col=mid,
            lo_col=lo,
            hi_col=hi,
            periods=PERIODS,
            title=f"Japan: {ttl} with CI band",
            out_png=fname,
            y_label="H sum",
        )

    print("\nSaved figures:")
    print(" - japan_H_pairwise_sums_CI_ALL_90.png")
    for _, _, _, _, fname in sum_panels:
        print(" -", fname)

    print("\nDone.")




