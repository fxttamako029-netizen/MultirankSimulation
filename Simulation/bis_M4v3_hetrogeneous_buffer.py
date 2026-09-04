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
    "2007Q1-2007Q4.csv",
    "2008Q1-2008Q4.csv",
    "2009Q1-2009Q4.csv",
]

POSITION = "C:Total claims"

LAYER_INSTRUMENTS = [
    ("All instruments", "A:All instruments"),          # Layer 0 (Core)
    ("Loans and deposits", "G:Loans and deposits"),    # Layer 1
]

TARGET_QUARTER = pd.Timestamp("2008-12-31") # Peak of crisis
VICTIM_COUNTRY = "US"  # Patient Zero

# --- 1. Heterogeneous Buffer Configuration ---
# structural assumption: Core hubs run on lower capital buffers (higher leverage).
# Periphery countries maintain higher buffers.

# List of Systemically Important Financial Centers (G10 + others)
CORE_COUNTRIES = [
    "US", "GB", "DE", "FR", "JP", "CH", "NL", "BE", "SE", "CA", "IT", "ES"
]

# Buffers for Tier 1 (Core/Hubs) - Fragile/High Leverage
BUFFER_LAYER_CORE = 0.03   # 3%
BUFFER_TOTAL_CORE = 0.05   # 5%

# Buffers for Tier 2 (Rest of World) - Robust/Lower Leverage
BUFFER_LAYER_REST = 0.08  # 8%
BUFFER_TOTAL_REST = 0.12   # 12%

# Structural Assumption 
EXTERNAL_ASSET_RATIO = 0.2

# =========================================================
# 2. DATA LOADING & PROCESSING
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
        return pd.read_csv(csv_path, encoding="utf-8-sig")

    return pd.read_csv(csv_path, skiprows=header_idx, encoding="utf-8-sig")

def code_and_name(label: str) -> tuple[str, str]:
    """Convert 'CH:Switzerland' -> ('CH', 'Switzerland')"""
    if isinstance(label, str) and ":" in label:
        c, n = label.split(":", 1)
        return c.strip(), n.strip()
    return str(label), str(label)

def build_nodes(df_all: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return sorted node labels, codes, and friendly names."""
    labels = set(df_all["L_REP_CTY:Reporting country"]).union(set(df_all["L_CP_COUNTRY:Counterparty country"]))
    labels = list(labels)
    # Sort by country code
    labels_sorted = sorted(labels, key=lambda s: code_and_name(s)[0])
    
    node_labels = labels_sorted
    node_codes = [code_and_name(s)[0] for s in node_labels] # We need codes for comparison
    node_names = [code_and_name(s)[1] for s in node_labels]
    
    return node_labels, node_codes, node_names

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
            
            if u == v: continue 
            if pd.isna(w) or float(w) <= 0: continue 
            
            if u in idx and v in idx:
                A[idx[u], idx[v]] = float(w)
                
        A_layers.append(A)
    return A_layers

# =========================================================
# 3. BUFFER ASSIGNMENT LOGIC
# =========================================================
def assign_heterogeneous_buffers(node_codes, N):
    """
    Creates buffer arrays where Core countries get lower thresholds 
    than Periphery countries.
    """
    b_vec = np.zeros(N)
    c_vec = np.zeros(N)
    
    print("\n--- Assigning Heterogeneous Buffers ---")
    core_count = 0
    
    for i, code in enumerate(node_codes):
        if code in CORE_COUNTRIES:
            b_vec[i] = BUFFER_LAYER_CORE
            c_vec[i] = BUFFER_TOTAL_CORE
            core_count += 1
        else:
            b_vec[i] = BUFFER_LAYER_REST
            c_vec[i] = BUFFER_TOTAL_REST
            
    print(f"Core Nodes (High Leverage/Low Buffer): {core_count}")
    print(f"Rest Nodes (Low Leverage/High Buffer): {N - core_count}")
    
    return b_vec, c_vec

# =========================================================
# 4. MULTIRANK ALGORITHM
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

        if np.allclose(x, x_new, atol=1e-6):
            x = x_new
            break

        x = x_new
        z = z_new

    return x, z

# =========================================================
# 5. MODIFIED: CONTAGION SIMULATION (With Asset Return)
# =========================================================
def simulate_contagion(
    A_layers, 
    initial_victim_idx, 
    b_buffer_vec, 
    c_buffer_vec, 
    external_asset_ratio=0.20, 
    max_steps=100
):
    L = len(A_layers)
    N = A_layers[0].shape[0]
    
    # 1. Calculate Network Assets 
    Assets_network = np.zeros((N, L))
    for alpha in range(L):
        Assets_network[:, alpha] = np.sum(A_layers[alpha], axis=1)
    
    # 2. Estimate TRUE Total Assets 
    Assets_total_estimated = np.sum(Assets_network, axis=1) / external_asset_ratio
    
    # Avoid division by zero for nodes with 0 assets (safe handling)
    Assets_total_estimated[Assets_total_estimated == 0] = 1.0
    
    Assets_layer_strict = Assets_network
    Assets_layer_strict[Assets_layer_strict == 0] = 1.0 
    
    # 3. Initialize State
    S = np.zeros((N, L), dtype=int) 
    
    if initial_victim_idx is not None:
        S[initial_victim_idx, :] = 1 
    
    # 4. Loop
    for t in range(max_steps):
        S_prev = S.copy()
        
        Loss_layer = np.zeros((N, L))
        for alpha in range(L):
            Loss_layer[:, alpha] = A_layers[alpha] @ S[:, alpha]
            
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_layer = Loss_layer / Assets_layer_strict 
            ratio_layer[np.isnan(ratio_layer)] = 0
            
        condition_horizontal = ratio_layer > b_buffer_vec[:, np.newaxis]
        
        Total_Loss = np.sum(Loss_layer, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_total = Total_Loss / Assets_total_estimated
            ratio_total[np.isnan(ratio_total)] = 0
        
        condition_vertical = ratio_total > c_buffer_vec
        
        S[condition_horizontal] = 1 
        for i in range(N):
            if condition_vertical[i]:
                S[i, :] = 1 
                
        if np.array_equal(S, S_prev):
            break
            
    # UPDATE: Now returning Assets_total_estimated as well
    return S, Assets_total_estimated

# =========================================================
# 6. VISUALIZATION FUNCTIONS 
# =========================================================
def plot_multirank_bars(scores, names, title, color='skyblue', top_n=15):
    df_scores = pd.DataFrame({'Name': names, 'Score': scores})
    df_scores = df_scores.sort_values(by='Score', ascending=True)
    
    if len(df_scores) > top_n:
        df_plot = df_scores.tail(top_n) 
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
# 7. MAIN EXECUTION
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
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
        frames.append(df)
    
    if not frames:
        print("No valid data loaded.")
        return
        
    df_all = pd.concat(frames, ignore_index=True)

    # 2. Build Network 
    node_labels, node_codes, node_names = build_nodes(df_all)
    if not node_labels:
        print("No nodes found.")
        return
        
    A_layers = build_A_layers_for_quarter(df_all, TARGET_QUARTER, node_labels)
    N = len(node_labels)
    
    print(f"--- Thesis Simulation: {TARGET_QUARTER.date()} ---")
    print(f"Total Nodes: {N}")
    
    # 3. PRE-CRISIS RANK
    x_pre, _ = multirank_iteration(A_layers)
    plot_multirank_bars(
        x_pre, node_names, 
        title=f"Pre-Crisis Rank ({TARGET_QUARTER.date()})",
        color='#1f77b4'
    )
    
    # 4. ASSIGN BUFFERS (Heterogeneous)
    b_vec, c_vec = assign_heterogeneous_buffers(node_codes, N)
    
    # 5. SHOCK SIMULATION
    print(f"\nSimulating Shock: {VICTIM_COUNTRY} Default with Heterogeneous Buffers...")
    
    try:
        victim_idx = [i for i, x in enumerate(node_labels) if x.startswith(VICTIM_COUNTRY)][0]
    except IndexError:
        print(f"Error: {VICTIM_COUNTRY} not found in dataset.")
        return

    # Pass Vectors to Simulation -> NOW CAPTURING ASSETS
    S_final, Assets_Est = simulate_contagion(
        A_layers, 
        victim_idx, 
        b_vec,  # Vector
        c_vec,  # Vector
        external_asset_ratio=EXTERNAL_ASSET_RATIO
    )
    
    # Identify Survivors
    # We define 'Dead' as failing in Layer 0 (All Instruments)
    is_dead = S_final[:, 0] == 1
    survivor_indices = np.where(~is_dead)[0]
    
    # ========================================================
    # NEW: SYSTEM ASSET LOSS CALCULATION
    # ========================================================
    total_system_assets = np.sum(Assets_Est)
    lost_assets = np.sum(Assets_Est[is_dead])
    
    # Avoid division by zero
    if total_system_assets > 0:
        pct_loss = (lost_assets / total_system_assets) * 100
    else:
        pct_loss = 0.0

    print("\n--- IMPACT ASSESSMENT ---")
    print(f"Patient Zero:      {VICTIM_COUNTRY}")
    print(f"Total Defaults:    {np.sum(is_dead)} / {N}")
    print(f"System Asset Loss: {pct_loss:.2f}%")
    print(f"Survivors:         {len(survivor_indices)}")
    
    if len(survivor_indices) < 2:
        print("System Collapse.")
        return

    # 6. POST-CRISIS RANK
    A_layers_post = []
    for A in A_layers:
        A_sub = A[np.ix_(survivor_indices, survivor_indices)]
        A_layers_post.append(A_sub)
        
    survivor_names = [node_names[i] for i in survivor_indices]
    
    x_post, _ = multirank_iteration(A_layers_post)
    
    plot_multirank_bars(
        x_post, survivor_names, 
        title=f"Post-Crisis Rank (Survivors)\nShock: {VICTIM_COUNTRY}",
        color='#ff7f0e'
    )

if __name__ == "__main__":
    main()