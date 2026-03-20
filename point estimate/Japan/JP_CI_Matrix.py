import pandas as pd
import numpy as np
import re

csv_path = "XXX/hurst_CI_field_bootstrap_JP_90.csv"   # <- change if needed

df = pd.read_csv(csv_path)

# Extract H-label (H1/H2/H3) from the "series" column
df["H_label"] = df["series"].str.extract(r"^(H\d)\b", expand=False)

# Keep only the 3 series we care about
df = df[df["H_label"].isin(["H1", "H2", "H3"])].copy()

# Desired order and labels for the table
order = ["H1", "H2", "H3"]
pretty = {
    "H1": "H1 (Sₜ)",
    "H2": "H2 (Vₜ)",
    "H3": "H3 (Rₜ)",
}

def make_sum_ci_table(period_df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Build a 3x3 table where entry (i,j) is the CI interval for H_i + H_j:
        [CI_lo(i)+CI_lo(j), CI_hi(i)+CI_hi(j)]
    Diagonal entries correspond to 2*H_i: [2*CI_lo(i), 2*CI_hi(i)].
    """
    # Ensure we have exactly one row per H1/H2/H3
    period_df = period_df.drop_duplicates(subset=["H_label"]).set_index("H_label").loc[order]
    if period_df.index.isnull().any() or len(period_df) != 3:
        raise ValueError("Each period must contain exactly one row for each of H1, H2, H3.")

    lo = period_df["CI_90_lo"].astype(float).to_numpy()
    hi = period_df["CI_90_hi"].astype(float).to_numpy()

    # Pairwise sums via broadcasting
    lo_sum = lo[:, None] + lo[None, :]
    hi_sum = hi[:, None] + hi[None, :]

    # Format as "[lower, upper]"
    fmt = lambda x: f"{x:.{decimals}f}"
    out = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            out[i, j] = f"[{fmt(lo_sum[i, j])}, {fmt(hi_sum[i, j])}]"

    table = pd.DataFrame(out, index=[pretty[h] for h in order], columns=[pretty[h] for h in order])
    return table

# Build tables for each period
tables = {}
for period, g in df.groupby("period", sort=False):
    tables[period] = make_sum_ci_table(g, decimals=4)

# Print tables
for period, tbl in tables.items():
    print("\n" + "=" * 80)
    print(period)
    print(tbl.to_string())




