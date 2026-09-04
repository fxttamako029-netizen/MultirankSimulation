import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
DATA_DIR = r"/Users/tamako029/Desktop/Honours Thesis/code"

CSV_FILES = [
    "2008Q1-2008Q4.csv", 
]

# Fixed narrative: exposure direction = Reporting -> Counterparty
POSITION = "C:Total claims"

# Layer definitions
LAYER_INSTRUMENTS = [
    ("All instruments", "A:All instruments"),          # Layer 0 (Core)
    ("Loans and deposits", "G:Loans and deposits"),    # Layer 1
]

# Analysis Settings
TARGET_QUARTER = pd.Timestamp("2008-12-31") # Peak of crisis
VICTIM_COUNTRY = "US"  # Patient Zero

# --- Thesis Simulation Parameters ---
# 1. Capital Buffers (Regulatory Constraints)
BUFFER_LAYER = 0.1   # b: buffer for specific asset class
BUFFER_TOTAL = 0.15   # c: buffer for total capital

# 2. Structural Assumption 
EXTERNAL_ASSET_RATIO = 0.2

# =========================================================
# 1. DATA LOADING & PROCESSING
# =========================================================
def read_bis_export(csv_path: str) -> pd.DataFrame:
    """Read BIS Data Portal CSV export (skips metadata lines)."""
    try:
        lines = Path(csv_path).read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return pd.DataFrame()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("DATAFLOW_ID:Dataflow ID"):
            header_idx = i
            break
    if header_idx is None:
        # Fallback: try to read directly if standard header missing
        return pd.read_csv(csv_path, encoding="utf-8-sig")

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
        # Filter data
        sub = df_all[
            (df_all["TIME_PERIOD:Period"] == quarter)
            & (df_all["L_POSITION:Balance sheet position"] == POSITION)
            & (df_all["L_INSTR:Type of instruments"] == instr)
        ]
        
        for _, r in sub.iterrows():
            u = r["L_REP_CTY:Reporting country"]
            v = r["L_CP_COUNTRY:Counterparty country"]
            w = r["OBS_VALUE:Value"]
            
            # Basic cleaning
            if u == v: continue 
            if pd.isna(w) or float(w) <= 0: continue 
            
            # Fill matrix
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

def multirank_iteration(A_layers, max_iter=1000, kappa=0.85, delta=1, gamma=1.0, phi=1):
    """Computes MultiRank centrality x (nodes) and z (layers)."""
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
        
        if np.sum(temp_z) == 0:
            z_new = np.ones(L) / L
        else:
            z_new = temp_z / np.sum(temp_z)

        x = x_new
        z = z_new

    return x, z

# =========================================================
# 3. CONTAGION SIMULATION
# =========================================================
def simulate_contagion(
    A_layers, 
    initial_victim_idx, 
    b_buffer, 
    c_buffer, 
    external_asset_ratio=0.20, 
    max_steps=100
):
    L = len(A_layers)
    N = A_layers[0].shape[0]
    
    # 1. Calculate Network Assets 
    Assets_network = np.zeros((N, L))
    for alpha in range(L):
        Assets_network[:, alpha] = np.sum(A_layers[alpha], axis=1) # a_i^net = sum_j A_ij
    
    # 2. Estimate TRUE Total Assets (Domestic + Network) 
    Assets_total_estimated = np.sum(Assets_network, axis=1) / external_asset_ratio # a_i^total = a_i^net / external
    Assets_layer_strict = Assets_network
    
    # 3. Initialize State
    S = np.zeros((N, L), dtype=int) 
    
    if initial_victim_idx is not None:
        S[initial_victim_idx, :] = 1 #set staus of initial victim to dead in all layers
    
    # 4. Loop
    for t in range(max_steps):
        S_prev = S.copy()
        
        Loss_layer = np.zeros((N, L))
        for alpha in range(L):
            Loss_layer[:, alpha] = A_layers[alpha] @ S[:, alpha] #loss(t) = sum_j A_ij * S_j(t))
            
        with np.errstate(divide='ignore', invalid='ignore'):#if denominator is 0, ignore warning
            ratio_layer = Loss_layer / Assets_layer_strict 
            ratio_layer[np.isnan(ratio_layer)] = 0
            
        condition_horizontal = ratio_layer > b_buffer
        
        Total_Loss = np.sum(Loss_layer, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_total = Total_Loss / Assets_total_estimated
            ratio_total[np.isnan(ratio_total)] = 0
            
        condition_vertical = ratio_total > c_buffer
        
        S[condition_horizontal] = 1 
        for i in range(N):
            if condition_vertical[i]:
                S[i, :] = 1 
                
        if np.array_equal(S, S_prev):
            break
            
    return S

# =========================================================
# 4. VISUALIZATION FUNCTIONS 
# =========================================================

def plot_multirank_bars(scores, names, title, color='skyblue', top_n=15):
    df_scores = pd.DataFrame({'Name': names, 'Score': scores})
    df_scores = df_scores.sort_values(by='Score', ascending=True) # Ascending for horizontal bar chart (top at top)
    
    if len(df_scores) > top_n:
        df_plot = df_scores.tail(top_n) # tail is top because we sorted ascending
    else:
        df_plot = df_scores

    plt.figure(figsize=(10, 6))
    
    plt.barh(df_plot['Name'], df_plot['Score'], color=color, edgecolor='black', alpha=0.7)
    
    plt.xlabel('MultiRank Score')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =========================================================
# 5. MAIN EXECUTION
# =========================================================
def main():
    # 1. Load Data
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"File not found: {path} (Skipping)")
            continue
        df = read_bis_export(path)
        if df.empty:
            print(f"Warning: {fname} is empty or unreadable.")
            continue
        # Convert Time Period
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
        frames.append(df)
    
    if not frames:
        print("No valid data loaded.")
        return
        
    df_all = pd.concat(frames, ignore_index=True)

    # 2. Build Network
    node_labels, node_names = build_nodes(df_all)
    if not node_labels:
        print("No nodes found.")
        return
        
    A_layers = build_A_layers_for_quarter(df_all, TARGET_QUARTER, node_labels)
    
    print(f"--- Thesis Simulation: {TARGET_QUARTER.date()} ---")
    print(f"Total Nodes: {len(node_labels)}")
    
    # ==========================================
    # PHASE 1: PRE-CRISIS
    # ==========================================
    x_pre, _ = multirank_iteration(A_layers)
    
    # Output: Pre-Crisis Bar Chart
    plot_multirank_bars(
        x_pre, 
        node_names, 
        title=f"Pre-Crisis MultiRank Centrality ({TARGET_QUARTER.date()})",
        color='#1f77b4', # Nice Blue
        top_n=15
    )
    
    # ==========================================
    # PHASE 2: SHOCK SIMULATION
    # ==========================================
    print(f"Parameters: Buffer={BUFFER_TOTAL}, Ext.Asset Ratio={EXTERNAL_ASSET_RATIO}")
    
    try:
        victim_idx = [i for i, x in enumerate(node_labels) if x.startswith(VICTIM_COUNTRY)][0]
    except IndexError:
        print(f"Error: {VICTIM_COUNTRY} not found in dataset.")
        return

    S_final = simulate_contagion(
        A_layers, 
        victim_idx, 
        BUFFER_LAYER, 
        BUFFER_TOTAL, 
        external_asset_ratio=EXTERNAL_ASSET_RATIO
    )
    
    # Identify Survivors (Alive in Layer 0)
    is_dead = S_final[:, 0] == 1
    survivor_indices = np.where(~is_dead)[0]
    
    print(f"--> Impact: {np.sum(is_dead)} defaults. {len(survivor_indices)} survivors.")
    
    if len(survivor_indices) < 2:
        print("System Collapse (Too few survivors to rank). Try increasing BUFFER or EXTERNAL_ASSET_RATIO.")
        return

    # ==========================================
    # PHASE 3: POST-CRISIS
    # ==========================================
    
    # Prune Network (Keep only survivors)
    A_layers_post = []
    for A in A_layers:
        A_sub = A[np.ix_(survivor_indices, survivor_indices)]
        A_layers_post.append(A_sub)
        
    survivor_names = [node_names[i] for i in survivor_indices]
    
    # Recalculate Rank
    x_post, _ = multirank_iteration(A_layers_post)
    
    # Output: Post-Crisis Bar Chart
    plot_multirank_bars(
        x_post, 
        survivor_names, 
        title=f"Post-Crisis MultiRank Centrality (Survivor Network)\nShock: {VICTIM_COUNTRY} Default",
        color='#ff7f0e',
        top_n=15
    )


if __name__ == "__main__":
    main()