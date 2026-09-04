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

POSITION = "C:Total claims"

LAYER_INSTRUMENTS = [
    ("All instruments", "A:All instruments"),          # Layer 0 (Core)
    ("Loans and deposits", "G:Loans and deposits"),    # Layer 1
]

TARGET_QUARTER = pd.Timestamp("2008-12-31") # Peak of crisis
VICTIM_COUNTRY = "US"  # Patient Zero
CRUNCH_RATIO = 0.5     # 50% Credit Withdrawal

# --- Heterogeneous Buffer Configuration ---
CORE_COUNTRIES = ["US", "GB", "DE", "FR", "JP", "CH", "NL", "BE", "SE", "CA", "IT", "ES"]

BUFFER_LAYER_CORE = 0.03   
BUFFER_TOTAL_CORE = 0.05   
BUFFER_LAYER_REST = 0.08   
BUFFER_TOTAL_REST = 0.12   

EXTERNAL_ASSET_RATIO = 0.2

# =========================================================
# 1. DATA LOADING & PROCESSING
# =========================================================
def read_bis_export(csv_path: str) -> pd.DataFrame:
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
    if isinstance(label, str) and ":" in label:
        c, n = label.split(":", 1)
        return c.strip(), n.strip()
    return str(label), str(label)

def build_nodes(df_all: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    labels = set(df_all["L_REP_CTY:Reporting country"]).union(set(df_all["L_CP_COUNTRY:Counterparty country"]))
    labels = list(labels)
    labels_sorted = sorted(labels, key=lambda s: code_and_name(s)[0])
    
    node_labels = labels_sorted
    node_codes = [code_and_name(s)[0] for s in node_labels]
    node_names = [code_and_name(s)[1] for s in node_labels]
    
    return node_labels, node_codes, node_names

def build_A_layers_for_quarter(df_all: pd.DataFrame, quarter: pd.Timestamp, node_labels: list[str]) -> list[np.ndarray]:
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
# 2. BUFFER ASSIGNMENT
# =========================================================
def assign_heterogeneous_buffers(node_codes, N):
    b_vec = np.zeros(N)
    c_vec = np.zeros(N)
    
    for i, code in enumerate(node_codes):
        if code in CORE_COUNTRIES:
            b_vec[i] = BUFFER_LAYER_CORE
            c_vec[i] = BUFFER_TOTAL_CORE
        else:
            b_vec[i] = BUFFER_LAYER_REST
            c_vec[i] = BUFFER_TOTAL_REST
    return b_vec, c_vec

# =========================================================
# 3. MULTIRANK ALGORITHM
# =========================================================
def multirank_iteration(A_layers, max_iter=1000, kappa=0.85):
    L = len(A_layers)
    N = A_layers[0].shape[0]
    W = np.zeros(L)
    B_in = np.zeros((L, N))

    for alpha in range(L):
        A = A_layers[alpha]
        W_alpha = np.sum(A)
        W[alpha] = W_alpha
        if W_alpha > 0:
            B_in[alpha, :] = np.sum(A, axis=0) / W_alpha

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
        
        # Simplified update for brevity
        if np.allclose(x, x_new, atol=1e-6):
            x = x_new
            break
        x = x_new

    return x, z

# =========================================================
# 4. CONTAGION WITH CREDIT CRUNCH (NEW)
# =========================================================
def simulate_contagion_with_crunch(
    A_layers, 
    initial_victim_idx, 
    b_buffer_vec, 
    c_buffer_vec, 
    external_asset_ratio=0.20, 
    crunch_ratio=0.0, # Default 0 = No Crunch. 0.5 = 50% withdrawal
    max_steps=100
):
    L = len(A_layers)
    N = A_layers[0].shape[0]
    
    # 1. Assets
    Assets_network = np.zeros((N, L))
    for alpha in range(L):
        Assets_network[:, alpha] = np.sum(A_layers[alpha], axis=1)
    
    Assets_total_estimated = np.sum(Assets_network, axis=1) / external_asset_ratio
    Assets_total_estimated[Assets_total_estimated == 0] = 1.0
    Assets_layer_strict = Assets_network
    Assets_layer_strict[Assets_layer_strict == 0] = 1.0 
    
    # 2. State
    S = np.zeros((N, L), dtype=int) 
    if initial_victim_idx is not None:
        S[initial_victim_idx, :] = 1 
    
    # --- NEW: CALCULATE FUNDING SHOCK (CREDIT CRUNCH) ---
    # Logic: If Victim (e.g., US) fails, it recalls 'crunch_ratio' of loans it made to others.
    # This creates an immediate "loss" (liquidity gap) for the borrowers.
    Funding_Shock = np.zeros((N, L))
    
    if initial_victim_idx is not None and crunch_ratio > 0:
        for alpha in range(L):
            # A[v, j] is what v (US) lent to j (Brazil).
            # US recalls this money. j loses this funding.
            # We treat this funding loss as equivalent to an asset loss for solvency check.
            loans_from_victim = A_layers[alpha][initial_victim_idx, :]
            Funding_Shock[:, alpha] = loans_from_victim * crunch_ratio
            
            # Note: The victim itself doesn't suffer funding shock from itself
            Funding_Shock[initial_victim_idx, alpha] = 0

    # 3. Loop
    for t in range(max_steps):
        S_prev = S.copy()
        
        Loss_layer = np.zeros((N, L))
        for alpha in range(L):
            # Normal Contagion: Loss from people who owe me money (A @ S)
            Solvency_Loss = A_layers[alpha] @ S[:, alpha]
            
            # Total Stress = Solvency Loss + Funding Shock (if victim is dead)
            # (Note: Funding shock is one-off, applied if S[victim] is 1. 
            # Since S[victim] starts at 1, it applies from t=0)
            
            # Check if victim is still dead (it should be)
            is_victim_dead = S[initial_victim_idx, alpha] == 1
            current_funding_shock = Funding_Shock[:, alpha] if is_victim_dead else 0
            
            Loss_layer[:, alpha] = Solvency_Loss + current_funding_shock
            
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
            
    return S, Assets_total_estimated

# =========================================================
# 5. VISUALIZATION
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
# 6. MAIN EXECUTION
# =========================================================
def main():
    # 1. Load Data
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        df = read_bis_export(path)
        if not df.empty:
            df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
            frames.append(df)
    
    if not frames:
        print("No valid data loaded.")
        return
    df_all = pd.concat(frames, ignore_index=True)

    # 2. Build Network
    node_labels, node_codes, node_names = build_nodes(df_all)
    A_layers = build_A_layers_for_quarter(df_all, TARGET_QUARTER, node_labels)
    N = len(node_labels)
    
    # 3. Buffers
    b_vec, c_vec = assign_heterogeneous_buffers(node_codes, N)
    
    try:
        victim_idx = node_codes.index(VICTIM_COUNTRY)
    except ValueError:
        print(f"Victim {VICTIM_COUNTRY} not found.")
        return

    # 4. SIMULATION 1: Normal Contagion (Solvency Only)
    print(f"\n--- Scenario A: Normal Contagion ({VICTIM_COUNTRY} Default) ---")
    S_normal, Assets_est = simulate_contagion_with_crunch(
        A_layers, victim_idx, b_vec, c_vec, EXTERNAL_ASSET_RATIO, 
        crunch_ratio=0.0 # No Crunch
    )
    dead_normal = np.sum(S_normal[:, 0] == 1)
    print(f"Defaults: {dead_normal}")

    # 5. SIMULATION 2: Credit Crunch (Solvency + Liquidity)
    print(f"\n--- Scenario B: With Credit Crunch ({int(CRUNCH_RATIO*100)}% Funding Withdrawal) ---")
    S_crunch, _ = simulate_contagion_with_crunch(
        A_layers, victim_idx, b_vec, c_vec, EXTERNAL_ASSET_RATIO, 
        crunch_ratio=CRUNCH_RATIO # With Crunch
    )
    dead_crunch = np.sum(S_crunch[:, 0] == 1)
    print(f"Defaults: {dead_crunch}")
    
    # 6. Compare
    new_victims_indices = [i for i in range(N) if S_normal[i, 0] == 0 and S_crunch[i, 0] == 1]
    
    print("\n" + "="*40)
    print("CREDIT CRUNCH IMPACT ANALYSIS")
    print("="*40)
    if new_victims_indices:
        print(f"Credit Crunch killed {len(new_victims_indices)} ADDITIONAL countries:")
        for i in new_victims_indices:
            print(f"- {node_names[i]} ({node_codes[i]})")
    else:
        print("No additional countries died.")
        print("(Likely because the 'Normal' shock was already too deadly, or buffers were high enough)")

    # 7. Post-Crisis Rank (Using Crunch Result)
    is_dead = S_crunch[:, 0] == 1
    survivor_indices = np.where(~is_dead)[0]
    
    if len(survivor_indices) > 1:
        A_layers_post = []
        for A in A_layers:
            A_sub = A[np.ix_(survivor_indices, survivor_indices)]
            A_layers_post.append(A_sub)
        survivor_names = [node_names[i] for i in survivor_indices]
        x_post, _ = multirank_iteration(A_layers_post)
        
        plot_multirank_bars(
            x_post, survivor_names, 
            title=f"Post-Crisis Rank (Survivors)\nShock: {VICTIM_COUNTRY} + Credit Crunch",
            color='#d62728' # Red for Crunch
        )

if __name__ == "__main__":
    main()