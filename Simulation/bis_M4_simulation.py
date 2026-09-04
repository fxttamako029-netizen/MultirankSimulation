import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Optional: Makes plots look better

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
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

# Analysis Settings
TARGET_QUARTER = pd.Timestamp("2008-12-31") # Focus on the peak of the crisis
VICTIM_COUNTRY = "US"  # The country that fails initially (Patient Zero)

# Contagion Parameters (Constant per thesis plan)
BUFFER_LAYER = 0.05  # b: Horizontal threshold 
BUFFER_TOTAL = 0.1 # c: Vertical threshold 

# =========================================================
# 1. DATA LOADING & PROCESSING
# =========================================================
def read_bis_export(csv_path: str) -> pd.DataFrame:
    """Read BIS Data Portal CSV export (skips metadata lines)."""
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
    """Convert 'CH:Switzerland' -> ('CH', 'Switzerland')"""
    if isinstance(label, str) and ":" in label:
        c, n = label.split(":", 1)
        return c.strip(), n.strip()
    return str(label), str(label)

def build_nodes(df_all: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return sorted node labels and friendly names."""
    labels = set(df_all["L_REP_CTY:Reporting country"]).union(set(df_all["L_CP_COUNTRY:Counterparty country"]))
    labels = list(labels)
    # Sort by country code
    labels_sorted = sorted(labels, key=lambda s: code_and_name(s)[0])
    node_labels = labels_sorted
    node_names = [code_and_name(s)[1] for s in node_labels]
    return node_labels, node_names

def build_A_layers_for_quarter(df_all: pd.DataFrame, quarter: pd.Timestamp, node_labels: list[str]) -> list[np.ndarray]:
    """Build multilayer adjacency matrices for a specific quarter."""
    N = len(node_labels)
    idx = {lab: i for i, lab in enumerate(node_labels)}
    A_layers = []

    for _, instr in LAYER_INSTRUMENTS:
        A = np.zeros((N, N), dtype=float)
        sub = df_all[
            (df_all["TIME_PERIOD:Period"] == quarter)
            & (df_all["L_POSITION:Balance sheet position"] == POSITION)
            & (df_all["L_INSTR:Type of instruments"] == instr)
        ]
        
        for _, r in sub.iterrows():
            u = r["L_REP_CTY:Reporting country"]
            v = r["L_CP_COUNTRY:Counterparty country"]
            w = r["OBS_VALUE:Value"]
            
            if u == v: continue # Ignore self-loops
            if pd.isna(w) or float(w) <= 0: continue # Ignore empty/negative
            
            # Exposure: Reporting (u) -> Counterparty (v)
            if u in idx and v in idx:
                A[idx[u], idx[v]] = float(w)
                
        A_layers.append(A)
    return A_layers

# =========================================================
# 2. MULTIRANK ALGORITHM
# =========================================================
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
    return W, B_in

def multirank_iteration(A_layers, max_iter=100, kappa=0.85, delta=1, gamma=1.0, phi=1):
    """
    Computes MultiRank centrality x (nodes) and z (layers).
    """
    L = len(A_layers)
    N = A_layers[0].shape[0]
    W, B_in = compute_W_and_B_in(A_layers)

    x = np.ones(N) / N
    z = np.ones(L) / L

    for _ in range(max_iter):
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
        
        # Handle case where all z become 0 (unlikely but possible)
        if np.sum(temp_z) == 0:
            z_new = np.ones(L) / L
        else:
            z_new = temp_z / np.sum(temp_z)

        x = x_new
        z = z_new

    return x, z

# =========================================================
# 3. CONTAGION SIMULATION (DOUBLE THRESHOLD)
# =========================================================
def simulate_contagion(A_layers, initial_victim_idx, b_buffer, c_buffer, max_steps=50):
    """
    Simulates financial contagion based on the Multiplex Threshold Model.
    Returns: Final state matrix S (N x L), where 1 = Distressed.
    """
    L = len(A_layers)
    N = A_layers[0].shape[0]
    
    # 1. Pre-compute Assets (Assumed as Row Sum of exposures)
    Assets_layer = np.zeros((N, L))
    for alpha in range(L):
        Assets_layer[:, alpha] = np.sum(A_layers[alpha], axis=1)
    
    Assets_total = np.sum(Assets_layer, axis=1)
    
    # 2. Initialize State
    S = np.zeros((N, L), dtype=int)
    
    # Initial Shock
    S[initial_victim_idx, :] = 1
    
    # 3. Simulation Loop
    for t in range(max_steps):
        S_prev = S.copy()
        
        # Calculate Loss per layer
        Loss_layer = np.zeros((N, L))
        for alpha in range(L):
            # If j is distressed (S[j]=1), i loses A[i,j]
            Loss_layer[:, alpha] = A_layers[alpha] @ S[:, alpha]
            
        # Check Horizontal Threshold (Layer specific)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_layer = Loss_layer / Assets_layer
            ratio_layer[np.isnan(ratio_layer)] = 0
            
        condition_horizontal = ratio_layer > b_buffer
        
        # Check Vertical Threshold (Total assets)
        Total_Loss = np.sum(Loss_layer, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_total = Total_Loss / Assets_total
            ratio_total[np.isnan(ratio_total)] = 0
            
        condition_vertical = ratio_total > c_buffer
        
        # Update State
        S[condition_horizontal] = 1 # Layer collapse
        
        # If vertical threshold met, collapse ALL layers for that node
        for i in range(N):
            if condition_vertical[i]:
                S[i, :] = 1
                
        # Check Convergence
        if np.array_equal(S, S_prev):
            break
            
    return S

# =========================================================
# 4. VISUALIZATION
# =========================================================
def plot_rank_comparison(df_cmp, quarter_date, victim_name):
    """
    Generates a scatter plot comparing Pre-Crisis vs Post-Crisis Ranks.
    Points below diagonal = Rank Improved.
    Points above diagonal = Rank Worsened.
    """
    # Set style for thesis (clean white background)
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot diagonal line (No change line)
    max_rank = max(df_cmp['Pre_Rank'].max(), df_cmp['Post_Rank'].max())
    ax.plot([0, max_rank], [0, max_rank], ls="--", c="gray", alpha=0.5, label="No Change")
    
    # Scatter plot
    # Color based on gain/loss
    colors = np.where(df_cmp['Rank_Change'] > 0, 'green', 
             np.where(df_cmp['Rank_Change'] < 0, 'red', 'blue'))
    
    scatter = ax.scatter(df_cmp['Pre_Rank'], df_cmp['Post_Rank'], 
                         c=colors, s=80, alpha=0.7, edgecolors='k')
    
    # Add labels for top movers or important countries
    # Label top 5 winners and top 5 losers
    top_winners = df_cmp.nlargest(5, 'Rank_Change')
    top_losers = df_cmp.nsmallest(5, 'Rank_Change')
    
    to_label = pd.concat([top_winners, top_losers])
    
    for _, row in to_label.iterrows():
        ax.text(row['Pre_Rank'] + 0.5, row['Post_Rank'], row['Country'], 
                fontsize=9, ha='left', va='center')

    # Formatting
    ax.set_title(f"Rank Stability Analysis: {victim_name} Shock ({quarter_date.date()})\n"
                 f"Parameters: b={BUFFER_LAYER}, c={BUFFER_TOTAL}", fontsize=14)
    ax.set_xlabel("Pre-Crisis Rank (1 is best)", fontsize=12)
    ax.set_ylabel("Post-Crisis Rank (1 is best)", fontsize=12)
    
    # Invert axes so Rank 1 is at top-right or adjust logic? 
    # Usually Rank 1 is close to origin (0,0) or top-left.
    # Let's keep 1 at bottom-left (standard xy) but invert axis visually is better for "Ranking"
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Rank Improved (Winner)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Rank Worsened (Loser)'),
        Line2D([0], [0], color='gray', lw=1, ls='--', label='Stability Line'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()

# =========================================================
# 5. MAIN EXECUTION
# =========================================================
def main():
    # Load Data
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = read_bis_export(path)
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    # Build Network
    node_labels, node_names = build_nodes(df_all)
    A_layers = build_A_layers_for_quarter(df_all, TARGET_QUARTER, node_labels)
    
    print(f"--- Analysis for Quarter: {TARGET_QUARTER.date()} ---")
    print(f"--- Simulating Shock: {VICTIM_COUNTRY} Collapse ---")

    # 1. Calculate PRE-CRISIS MultiRank
    x_pre, _ = multirank_iteration(A_layers)
    
    # 2. Run Contagion
    try:
        victim_idx = [i for i, x in enumerate(node_labels) if x.startswith(VICTIM_COUNTRY)][0]
    except IndexError:
        print(f"Error: Country code {VICTIM_COUNTRY} not found in data.")
        return

    S_final = simulate_contagion(A_layers, victim_idx, BUFFER_LAYER, BUFFER_TOTAL)
    
    # 3. Identify Survivors
    # Definition: A country survives if it is NOT distressed in Layer 0 (Core layer)
    # Or strict definition: NOT distressed in ANY layer. Let's use Layer 0.
    is_dead = S_final[:, 0] == 1
    survivor_indices = np.where(~is_dead)[0]
    
    print(f"Total Countries: {len(node_labels)}")
    print(f"Defaults: {np.sum(is_dead)} (including {VICTIM_COUNTRY})")
    print(f"Survivors: {len(survivor_indices)}")
    
    if len(survivor_indices) < 2:
        print("System collapse. Cannot compute post-rank.")
        return

    # 4. Prune Network (Create Post-Crisis Network)
    A_layers_post = []
    for A in A_layers:
        A_sub = A[np.ix_(survivor_indices, survivor_indices)]
        A_layers_post.append(A_sub)
        
    # 5. Calculate POST-CRISIS MultiRank
    x_post, _ = multirank_iteration(A_layers_post)
    
    # 6. Compare Ranks
    # We must compare the rank OF THE SURVIVORS among themselves before and after.
    
    survivor_names = [node_names[i] for i in survivor_indices]
    
    # Get Pre-Scores of survivors only
    x_pre_survivors = x_pre[survivor_indices]
    
    # Calculate Ranks (descending score -> Rank 1 is best)
    # method='min' handles ties
    rank_pre = pd.Series(x_pre_survivors).rank(ascending=False, method='min')
    rank_post = pd.Series(x_post).rank(ascending=False, method='min')
    
    df_cmp = pd.DataFrame({
        'Country': survivor_names,
        'Pre_Rank': rank_pre.values,
        'Post_Rank': rank_post.values
    })
    
    # Calculate Change: Positive means Rank Improved (Numerical value decreased)
    # e.g. Rank 10 -> Rank 5. Change = 10 - 5 = +5 (Improvement)
    df_cmp['Rank_Change'] = df_cmp['Pre_Rank'] - df_cmp['Post_Rank']
    
    print("\n--- Top 5 Winners (Rank Improved) ---")
    print(df_cmp.nlargest(5, 'Rank_Change').to_string(index=False))
    
    print("\n--- Top 5 Losers (Rank Worsened) ---")
    print(df_cmp.nsmallest(5, 'Rank_Change').to_string(index=False))
    
    # 7. Plot
    plot_rank_comparison(df_cmp, TARGET_QUARTER, VICTIM_COUNTRY)

if __name__ == "__main__":
    main()