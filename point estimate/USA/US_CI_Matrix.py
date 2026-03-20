import pandas as pd
import numpy as np
import re

# Input bootstrap-CI table
# ============================================================
# Purpose:
#   This script reads the CSV file produced by US_Confidence_Interval.py.
#   That CSV contains, for each period and for each of the three U.S. series,
#   the estimated Hurst parameter H_i together with its bootstrap confidence
#   interval.
#
# Goal of the present script:
#   Convert those marginal confidence intervals into 3x3 matrices of
#   confidence intervals for the pairwise sums
#
#       H_i + H_j.
#
# These sums are the central diagnostics for cross long-term memory:
#   - if H_i + H_j > 1, the corresponding pair is associated with cross
#     long-term memory;
#   - if H_i + H_j = 1, the pair lies on the critical boundary;
#   - if H_i + H_j < 1, cross long-term memory is absent.


csv_path = "XXX/hurst_CI_field_bootstrap_US_90.csv"  # <- change if needed

df = pd.read_csv(csv_path)

# Extract the symbolic label H1 / H2 / H3 from the descriptive series names.
#Then index the rows by 'H1, H2, H3' to access rows by label later
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

# CI matrix construction
# ============================================================
# Main idea:
#   Suppose that for a fixed period we already have marginal confidence intervals:
#   H_i in [L_i, U_i],   i = 1,2,3.
#
#   Then the script forms the interval: 'H_i + H_j' in [L_i + L_j, U_i + U_j].
#
# Structure of the table:
#   - Off-diagonal entry (i,j): CI for H_i + H_j
#   - Diagonal entry   (i,i): CI for 2H_i
#
# Because H_i + H_j = H_j + H_i, the matrix is symmetric.
#
# Interpretation:
#   - Entire interval above 1  -> evidence supporting cross long-term memory
#   - Interval straddling 1    -> inconclusive / near-critical case
#   - Entire interval below 1  -> no cross long-term memory

def make_sum_ci_table_90(period_df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Build a 3x3 confidence-interval table for the sums H_i + H_j.

    Parameters:
        period_df : Subtable corresponding to a single period.
        decimals : Number of decimal places used in the printed output.

    Returns:
        A symmetric 3x3 matrix whose (i,j)-entry is the interval
        [CI_lo(i)+CI_lo(j), CI_hi(i)+CI_hi(j)].
    """

    # Ensure that the current period contains exactly one row for each of
    # H1, H2, and H3, and reorder them in the desired canonical order.
    period_df = period_df.drop_duplicates(subset=["H_label"]).set_index("H_label").loc[order]
    if period_df.index.isnull().any() or len(period_df) != 3:
        raise ValueError("Each period must contain exactly one row for each of H1, H2, H3.")

    # Extract lower and upper bounds of the marginal confidence intervals.
    lo = period_df["CI_90_lo"].astype(float).to_numpy()
    hi = period_df["CI_90_hi"].astype(float).to_numpy()

    # Pairwise sums via broadcasting:
    #   lo_sum[i,j] = lo[i] + lo[j]
    #   hi_sum[i,j] = hi[i] + hi[j]
    lo_sum = lo[:, None] + lo[None, :]
    hi_sum = hi[:, None] + hi[None, :]

    # Format as "[lower, upper]"
    fmt = lambda x: f"{x:.{decimals}f}"
    out = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            out[i, j] = f"[{fmt(lo_sum[i, j])}, {fmt(hi_sum[i, j])}]"

    # Attach readable row and column labels.
    table = pd.DataFrame(out, index=[pretty[h] for h in order], columns=[pretty[h] for h in order])
    return table

# Build one CI matrix for each period
# ============================================================
# The CSV file is in long format, so we group by period and then apply the
# table-construction function above to each block separately.

tables = {}
for period, g in df.groupby("period", sort=False):
    tables[period] = make_sum_ci_table_90(g, decimals=4)

# Print the resulting tables
# ============================================================
# Each printed matrix summarizes, for one sub-period, the confidence
# intervals of the nine quantities:
#
#   2H1,     H1+H2,   H1+H3,
#   H2+H1,   2H2,     H2+H3,
#   H3+H1,   H3+H2,   2H3.
#
# In this paper, these tables are used to determine which pairs of
# financial variables exhibit statistically supported cross long-term memory.
for period, tbl in tables.items():
    print("\n" + "=" * 80)
    print(period)
    print(tbl.to_string())





