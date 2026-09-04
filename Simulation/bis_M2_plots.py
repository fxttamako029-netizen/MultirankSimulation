import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Milestone 2 plots (single-layer, exposure direction)
# Exposure direction: Reporting country -> Counterparty country
# =========================================================

# 1) Your local file locations
DATA_DIR = r"/Users/tamako029/Desktop/Honours Thesis/code"
CSV_FILES = [
    "2006Q1-2007Q2.csv",
    "2007Q3-2008Q4.csv",
    "2009Q1-2009Q4.csv",
]

# 2) Choose which slice to visualize (keep fixed for comparability)
POSITION = "C:Total claims"         # or "L:Total liabilities"
INSTRUMENT = "A:All instruments"    # or "G:Loans and deposits"

# 3) Output / display
SHOW_PLOTS = True
SAVE_PLOTS = False

# 4) Representative quarter rule for each CSV segment
#    (simple choice: take the last quarter in the file)
REP_QUARTER_RULE = "last"  # keep as "last" for now


def read_bis_export(csv_path: str) -> pd.DataFrame:
    """Read BIS Data Portal CSV export (skips metadata lines above the true header)."""
    lines = Path(csv_path).read_text(errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("DATAFLOW_ID:Dataflow ID"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot find BIS header row in: {csv_path}")

    return pd.read_csv(csv_path, skiprows=header_idx, encoding="utf-8-sig")


def country_code(label: str) -> str:
    """Convert 'US:United States' -> 'US'."""
    return label.split(":", 1)[0] if isinstance(label, str) else str(label)


def pick_representative_quarter(df: pd.DataFrame) -> pd.Timestamp:
    quarters = sorted(df["TIME_PERIOD:Period"].dropna().unique())
    if not quarters:
        raise ValueError("No quarters available after filtering.")
    return quarters[-1]


def build_adjacency(df_q: pd.DataFrame, nodes: list[str]) -> np.ndarray:
    """Build adjacency matrix A where A[i,j] is exposure from i -> j."""
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=float)

    for _, r in df_q.iterrows():
        u = r["L_REP_CTY:Reporting country"]
        v = r["L_CP_COUNTRY:Counterparty country"]
        if u == v:
            continue

        w = r["OBS_VALUE:Value"]
        # NOTE: BIS exports may contain missing values (suppression/coverage).
        # For these basic plots we treat missing as 0 to keep matrices well-defined.
        if pd.isna(w) or float(w) <= 0:
            continue

        A[idx[u], idx[v]] = float(w)

    return A


def save_or_show(fig, filename: str):
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def main():
    # -----------------------------------------------------
    # Load + filter all data (single-layer slice)
    # -----------------------------------------------------
    all_df = []
    segment_dfs = []  # keep each CSV separately for representative snapshots

    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = read_bis_export(path)
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")

        df = df[
            (df["L_POSITION:Balance sheet position"] == POSITION)
            & (df["L_INSTR:Type of instruments"] == INSTRUMENT)
        ].copy()

        df["segment_file"] = fname
        all_df.append(df)
        segment_dfs.append((fname, df))

    df_all = pd.concat(all_df, ignore_index=True)

    # Node set (G10) from the data itself
    nodes = sorted(
        set(df_all["L_REP_CTY:Reporting country"]).union(set(df_all["L_CP_COUNTRY:Counterparty country"]))
    )
    node_codes = [country_code(n) for n in nodes]

    # -----------------------------------------------------
    # Plot 1: Total exposure over time (sum of all edges)
    # -----------------------------------------------------
    totals = (
        df_all.groupby("TIME_PERIOD:Period", as_index=False)["OBS_VALUE:Value"]
        .sum(min_count=1)
        .sort_values("TIME_PERIOD:Period")
    )

    fig = plt.figure(figsize=(10, 5))
    plt.plot(totals["TIME_PERIOD:Period"], totals["OBS_VALUE:Value"], marker="o")
    plt.title(f"Total cross-border exposure over time\n{POSITION} | {INSTRUMENT}")
    plt.xlabel("Quarter")
    plt.ylabel("Total exposure (USD, millions)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_or_show(fig, "total_exposure_over_time.png")

if __name__ == "__main__":
    main()
