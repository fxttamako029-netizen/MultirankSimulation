import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_plot():
    """Show the current figure and wait until the window is closed."""
    plt.show()


# =========================================================
# Milestone 3 (BIS -> multilayer A_layers -> MultiRank)
# Layer 0 = Claims + All instruments
# Layer 1 = Claims + Loans and deposits
# =========================================================

# -------------------------------
# 0) File paths and layer definition
# -------------------------------
DATA_DIR = r"/Users/tamako029/Desktop/Honours Thesis/code"
CSV_FILES = [
    "2007Q1-2007Q4.csv",
    "2008Q1-2008Q4.csv",
    "2009Q1-2009Q4.csv",
]

# Fixed narrative: exposure direction = Reporting -> Counterparty, using Claims
POSITION = "C:Total claims"

# Layer definitions (instrument types)
LAYER_INSTRUMENTS = [
    ("All instruments", "A:All instruments"),          # Layer 0
    ("Loans and deposits", "G:Loans and deposits"),    # Layer 1
]

# Representative quarters to run (Pre / Crisis / Post)
REP_QUARTERS = [
    pd.Timestamp("2007-06-30"),
    pd.Timestamp("2008-12-31"),
    pd.Timestamp("2009-12-31"),
]


# -------------------------------
# 1) Read BIS export CSV (skip metadata header)
# -------------------------------
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


def code_and_name(label: str) -> tuple[str, str]:
    """
    Convert 'CH:Switzerland' -> ('CH', 'Switzerland')
    If no ':', return (label, label)
    """
    if isinstance(label, str) and ":" in label:
        c, n = label.split(":", 1)
        return c.strip(), n.strip()
    return str(label), str(label)


# -------------------------------
# 2) Build multilayer adjacency matrices A_layers for one quarter
# -------------------------------
def build_nodes(df_all: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return:
      node_labels: list like ['BE:Belgium', 'CA:Canada', ...] in alphabetical code order
      node_names : list like ['Belgium', 'Canada', ...] aligned with node_labels
    """
    labels = set(df_all["L_REP_CTY:Reporting country"]).union(set(df_all["L_CP_COUNTRY:Counterparty country"]))
    labels = list(labels)

    # sort by country code so ordering is stable and alphabetic
    labels_sorted = sorted(labels, key=lambda s: code_and_name(s)[0])

    node_labels = labels_sorted
    node_names = [code_and_name(s)[1] for s in node_labels]
    return node_labels, node_names


def build_A_for_quarter_and_layer(
    df_all: pd.DataFrame,
    quarter: pd.Timestamp,
    position: str,
    instrument: str,
    node_labels: list[str],
) -> np.ndarray:
    """
    Build one N x N adjacency matrix A for a specific quarter + layer.
    A[i,j] = exposure from Reporting i -> Counterparty j
    Missing/suppressed values are treated as 0 (i.e., no edge added).
    """
    N = len(node_labels)
    idx = {lab: i for i, lab in enumerate(node_labels)}
    A = np.zeros((N, N), dtype=float) #N x N adjacency matrix

    sub = df_all[
        (df_all["TIME_PERIOD:Period"] == quarter)
        & (df_all["L_POSITION:Balance sheet position"] == position)
        & (df_all["L_INSTR:Type of instruments"] == instrument)
    ].copy()

    for _, r in sub.iterrows(): #iterate through the table row by row
        u = r["L_REP_CTY:Reporting country"]
        v = r["L_CP_COUNTRY:Counterparty country"]
        if u == v:
            continue  # cross-border focus; drop self-loop

        w = r["OBS_VALUE:Value"]
        if pd.isna(w) or float(w) <= 0:
            continue  # suppression/missing treated as 0

        A[idx[u], idx[v]] = float(w)

    return A


def build_A_layers_for_quarter(df_all: pd.DataFrame, quarter: pd.Timestamp, node_labels: list[str]) -> list[np.ndarray]:
    """Return A_layers = [A_layer0, A_layer1] for the given quarter."""
    A_layers = []
    for _layer_name, instr in LAYER_INSTRUMENTS:
        A = build_A_for_quarter_and_layer(df_all, quarter, POSITION, instr, node_labels)
        A_layers.append(A)
    return A_layers


# -------------------------------
# 3) MultiRank helpers (KEEP FORMULA UNCHANGED)
# -------------------------------
def compute_W_and_B_in(A_layers): #total weight per layer and normalized in-strength per node per layer
    L = len(A_layers) 
    N = A_layers[0].shape[0]   #number of nodes

    W = np.zeros(L) # total weight per layer
    B_in = np.zeros((L, N)) # normalized in-strength per node per layer

    for alpha in range(L): 
        A = A_layers[alpha]
        W_alpha = np.sum(A)  
        W[alpha] = W_alpha  

        if W_alpha > 0: 
            in_strength = np.sum(A, axis=0)
            B_in[alpha, :] = in_strength / W_alpha
        else:
            B_in[alpha, :] = 0.0 

    return W, B_in


def multirank_iteration(
    A_layers,
    max_iter=100,
    kappa=0.85,
    delta=1,
    gamma=1.0,
    phi=1
):
    L = len(A_layers)
    N = A_layers[0].shape[0]

    W, B_in = compute_W_and_B_in(A_layers)

    x = np.ones(N) / N  # all nodes equal initially
    z = np.ones(L) / L  # all layers equal initially

    x_hist = [x.copy()]
    z_hist = [z.copy()]

    for it in range(1, max_iter + 1):

        # (1) Compute weighted adjacency matrix G_ij
        G = np.zeros((N, N))
        for alpha in range(L):
            G += z[alpha] * A_layers[alpha]

        # (2) Compute out-strength k_j
        k = np.sum(G, axis=1)
        k_safe = k.copy()
        k_safe[k_safe == 0] = 1.0

        RW = np.transpose(G) @ (x / k_safe)  # Random Walk step (try transpose)

        x_new = kappa * RW + (1 - kappa) / N
        x_new /= np.sum(x_new)

        # (3) Update z_alpha
        temp_z = np.zeros(L)
        for alpha in range(L):
            s_alpha = np.sum(B_in[alpha, :] * (x_new ** gamma))
            temp_z[alpha] = (W[alpha] ** phi) * (s_alpha ** delta)

        z_new = temp_z / np.sum(temp_z)

        x = x_new
        z = z_new

        x_hist.append(x.copy())
        z_hist.append(z.copy())

        print(f"Iteration {it}: z = {z}")

    return np.array(x_hist), np.array(z_hist)


# -------------------------------
# 4) Run MultiRank for 3 representative quarters (output full country names)
# -------------------------------
def main():
    # Load and concat all three CSV files
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = read_bis_export(path)
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    # Build nodes (alphabetical by country code), but output full names
    node_labels, node_names = build_nodes(df_all)

    # Quick sanity: show node list as names
    print("\nNode order (alphabetical codes) -> full names:")
    for lab, name in zip(node_labels, node_names):
        print(f"  {code_and_name(lab)[0]} -> {name}")

    # Run each representative quarter
    for quarter in REP_QUARTERS:
        print("\n" + "=" * 60)
        print(f"Running MultiRank for quarter: {quarter.date()}")
        print("Layers:")
        for i, (lname, _) in enumerate(LAYER_INSTRUMENTS):
            print(f"  Layer {i}: {lname}")

        A_layers = build_A_layers_for_quarter(df_all, quarter, node_labels)

        # Run MultiRank (formulas unchanged)
        x_hist, z_hist = multirank_iteration(A_layers, max_iter=30)

        x_final = x_hist[-1]
        z_final = z_hist[-1]

        # Show final layer influence
        print("\nFinal layer scores z:")
        for i, (lname, _) in enumerate(LAYER_INSTRUMENTS):
            print(f"  {lname}: {z_final[i]:.6f}")

        # Show final node scores (top to bottom)
        order = np.argsort(-x_final)
        print("\nFinal node scores x (sorted):")
        for rank, idx in enumerate(order, start=1):
            print(f"  {rank:2d}. {node_names[idx]}  x={x_final[idx]:.6f}")

        # Simple visualization: node scores bar chart (full names)
        plt.figure(figsize=(12, 5))
        plt.bar(node_names, x_final)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Node MultiRank score x")
        plt.title(f"Node scores (x) — {quarter.date()}")
        plt.tight_layout()
        show_plot()

        # Layer score evolution plot
        iters = np.arange(z_hist.shape[0])
        plt.figure(figsize=(7, 5))
        for i, (lname, _) in enumerate(LAYER_INSTRUMENTS):
            plt.plot(iters, z_hist[:, i], marker="o", label=lname)
        plt.xlabel("Iteration")
        plt.ylabel(r"$z_{\alpha}$ (layer MultiRank)")
        plt.title(f"Layer score evolution — {quarter.date()}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        show_plot()


if __name__ == "__main__":
    main()