import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    """
    N = len(node_labels)
    idx = {lab: i for i, lab in enumerate(node_labels)}
    A = np.zeros((N, N), dtype=float)

    sub = df_all[
        (df_all["TIME_PERIOD:Period"] == quarter)
        & (df_all["L_POSITION:Balance sheet position"] == position)
        & (df_all["L_INSTR:Type of instruments"] == instrument)
    ].copy()

    for _, r in sub.iterrows():
        u = r["L_REP_CTY:Reporting country"]
        v = r["L_CP_COUNTRY:Counterparty country"]
        if u == v:
            continue  # cross-border focus; drop self-loop

        w = r["OBS_VALUE:Value"]
        if pd.isna(w) or float(w) <= 0:
            continue

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
def compute_W_and_B_in(A_layers): 
    L = len(A_layers) 
    N = A_layers[0].shape[0]

    W = np.zeros(L) 
    B_in = np.zeros((L, N)) 

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

    x = np.ones(N) / N 
    z = np.ones(L) / L 

    x_hist = [x.copy()]
    z_hist = [z.copy()]

    for it in range(1, max_iter + 1):
        G = np.zeros((N, N))
        for alpha in range(L):
            G += z[alpha] * A_layers[alpha]

        k = np.sum(G, axis=1)
        k_safe = k.copy()
        k_safe[k_safe == 0] = 1.0

        RW = np.transpose(G) @ (x / k_safe) 

        x_new = kappa * RW + (1 - kappa) / N
        x_new /= np.sum(x_new)

        temp_z = np.zeros(L)
        for alpha in range(L):
            s_alpha = np.sum(B_in[alpha, :] * (x_new ** gamma))
            temp_z[alpha] = (W[alpha] ** phi) * (s_alpha ** delta)

        z_new = temp_z / np.sum(temp_z)

        x = x_new
        z = z_new

        x_hist.append(x.copy())
        z_hist.append(z.copy())

    return np.array(x_hist), np.array(z_hist)


# -------------------------------
# 4) 3D Multiplex Visualization (Black & Gold)
# -------------------------------
def draw_3d_empirical_multiplex(A_layers, node_names, quarter, z_final, top_k=40):
    """
    Draws a 3D multiplex network using a circular layout.
    Filters to show only the Top-K edges per layer to avoid visual clutter.
    """
    plt.style.use('dark_background')
    STYLE = {
        'bg': 'black', 'node': '#FFD700', 
        'e0': '#00FFFF', # Cyan for Layer 0 (Total Claims)
        'e1': '#FF3333', # Red for Layer 1 (Loans & Deposits)
        'font': 'white'
    }
    
    N = len(node_names)
    fig = plt.figure(figsize=(10, 8), facecolor=STYLE['bg'])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(STYLE['bg'])
    ax.axis('off')

    # Create a circular layout for nodes
    dummy_G = nx.cycle_graph(N)
    pos_2d = nx.circular_layout(dummy_G)
    
    z0, z1_h = 0, 2
    x_coords = [pos_2d[i][0] for i in range(N)]
    y_coords = [pos_2d[i][1] for i in range(N)]

    # Draw nodes for both layers
    ax.scatter(x_coords, y_coords, [z0]*N, c=STYLE['node'], s=150, alpha=0.9, edgecolors='white')
    ax.scatter(x_coords, y_coords, [z1_h]*N, c=STYLE['node'], s=150, alpha=0.9, edgecolors='white')

    # Draw Inter-layer dashed lines (connecting the same country)
    for i in range(N):
        ax.plot([x_coords[i], x_coords[i]], [y_coords[i], y_coords[i]], [z0, z1_h], 
                color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Add labels (only to top layer to avoid clutter)
        # Convert full name to 3-letter code or keep short for display if preferred, here using full name
        short_name = node_names[i][:3].upper() # using first 3 letters for neatness
        ax.text(x_coords[i]*1.1, y_coords[i]*1.1, z1_h, short_name, color=STYLE['font'], fontsize=8, ha='center')

    # Helper function to get top edges
    def get_top_edges(A, k):
        edges = []
        for i in range(N):
            for j in range(N):
                if A[i, j] > 0:
                    edges.append((i, j, A[i, j]))
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges[:k]

    # Draw Edges Layer 0
    top_edges_L0 = get_top_edges(A_layers[0], top_k)
    max_w0 = max([w for _, _, w in top_edges_L0]) if top_edges_L0 else 1
    for u, v, w in top_edges_L0:
        lw = 0.5 + 2.5 * (w / max_w0) # scale line width
        ax.plot([x_coords[u], x_coords[v]], [y_coords[u], y_coords[v]], [z0, z0], 
                color=STYLE['e0'], alpha=0.6, lw=lw)

    # Draw Edges Layer 1
    top_edges_L1 = get_top_edges(A_layers[1], top_k)
    max_w1 = max([w for _, _, w in top_edges_L1]) if top_edges_L1 else 1
    for u, v, w in top_edges_L1:
        lw = 0.5 + 2.5 * (w / max_w1)
        ax.plot([x_coords[u], x_coords[v]], [y_coords[u], y_coords[v]], [z1_h, z1_h], 
                color=STYLE['e1'], alpha=0.6, lw=lw)

    # Layer Titles
    ax.text(1.2, 0, z0, f"Layer 0: Total Claims\n(z={z_final[0]:.2f})", color=STYLE['e0'], fontsize=11, fontweight='bold')
    ax.text(1.2, 0, z1_h, f"Layer 1: Loans & Deposits\n(z={z_final[1]:.2f})", color=STYLE['e1'], fontsize=11, fontweight='bold')
    
    # View angle
    ax.view_init(elev=20, azim=-45)
    plt.title(f"Empirical BIS Multiplex Network\n{quarter.date()} (Top {top_k} Edges shown)", color=STYLE['font'], fontsize=14, pad=20)
    plt.tight_layout()


# -------------------------------
# 5) Main execution
# -------------------------------
def main():
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = read_bis_export(path)
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    node_labels, node_names = build_nodes(df_all)

    print("\nNode order (alphabetical codes) -> full names:")
    for lab, name in zip(node_labels, node_names):
        print(f"  {code_and_name(lab)[0]} -> {name}")

    for quarter in REP_QUARTERS:
        print("\n" + "=" * 60)
        print(f"Running MultiRank for quarter: {quarter.date()}")
        
        A_layers = build_A_layers_for_quarter(df_all, quarter, node_labels)
        x_hist, z_hist = multirank_iteration(A_layers, max_iter=30)

        x_final = x_hist[-1]
        z_final = z_hist[-1]

        # 1. Show the 3D Multiplex Network (NEW FEATURE)
        print(f"Generating 3D Multiplex Visualization for {quarter.date()}...")
        draw_3d_empirical_multiplex(A_layers, node_names, quarter, z_final, top_k=40)
        show_plot()

        # 2. Node scores bar chart
        plt.figure(figsize=(12, 5))
        plt.bar(node_names, x_final, color='#FFD700') # added gold color to match theme
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Node MultiRank score x")
        plt.title(f"Node scores (x) — {quarter.date()}")
        plt.tight_layout()
        show_plot()

        # 3. Layer score evolution plot
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