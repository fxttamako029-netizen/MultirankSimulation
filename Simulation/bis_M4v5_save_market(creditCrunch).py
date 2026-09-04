import os
from pathlib import Path
import numpy as np
import pandas as pd

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
            
            if u == v or pd.isna(w) or float(w) <= 0: continue 
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
# 3. UPGRADED: CONTAGION + CRUNCH + MULTI-BAILOUT
# =========================================================
def simulate_contagion_full_scenario(
    A_layers, 
    initial_victim_indices, # NOW A LIST
    b_buffer_vec, 
    c_buffer_vec, 
    external_asset_ratio=0.20, 
    crunch_ratio=0.0, 
    bailout_indices=None,   # NOW A LIST
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
    Assets_layer_strict = Assets_network.copy()
    Assets_layer_strict[Assets_layer_strict == 0] = 1.0 
    
    # 2. State Initialization
    S = np.zeros((N, L), dtype=int) 
    if initial_victim_indices:
        for v_idx in initial_victim_indices:
            S[v_idx, :] = 1 
        
    # **CRITICAL**: If we bailout an initial victim immediately, they never die.
    if bailout_indices:
        for b_idx in bailout_indices:
            if b_idx in initial_victim_indices:
                S[b_idx, :] = 0

    # 3. Loop
    for t in range(max_steps):
        S_prev = S.copy()
        Loss_layer = np.zeros((N, L))
        
        # Calculate dynamic funding shock (Only dead victims trigger credit crunch)
        current_funding_shock = np.zeros((N, L))
        if initial_victim_indices and crunch_ratio > 0:
            for alpha in range(L):
                for v_idx in initial_victim_indices:
                    if S[v_idx, alpha] == 1: # Victim is DEAD
                        current_funding_shock[:, alpha] += A_layers[alpha][v_idx, :] * crunch_ratio

        for alpha in range(L):
            # Solvency Loss + Liquidity Loss
            Solvency_Loss = A_layers[alpha] @ S[:, alpha]
            Loss_layer[:, alpha] = Solvency_Loss + current_funding_shock[:, alpha]
            
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
        
        # --- APPLY INTERVENTION (Resurrect / Shield) ---
        if bailout_indices:
            for b_idx in bailout_indices:
                S[b_idx, :] = 0
                
        if np.array_equal(S, S_prev):
            break
            
    return S, Assets_total_estimated

# =========================================================
# 4. MAIN EXECUTION (EXPERIMENT MATRIX)
# =========================================================
def main():
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = read_bis_export(path)
        if not df.empty:
            df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
            frames.append(df)
    
    if not frames:
        print("No valid data loaded.")
        return
    df_all = pd.concat(frames, ignore_index=True)

    node_labels, node_codes, node_names = build_nodes(df_all)
    A_layers = build_A_layers_for_quarter(df_all, TARGET_QUARTER, node_labels)
    N = len(node_labels)
    b_vec, c_vec = assign_heterogeneous_buffers(node_codes, N)
    
    # ---------------------------------------------------------
    # DEFINE THE COMPREHENSIVE EXPERIMENT SET
    # Format: {"category": "...", "shock": ["US"], "bailout": ["GB"]}
    # ---------------------------------------------------------
    experiments = [
        # --- US Shock ---
        {"cat": "1. US Shock", "shock": ["US"], "bailout": ["US"]},
        {"cat": "1. US Shock", "shock": ["US"], "bailout": ["GB"]},
        {"cat": "1. US Shock", "shock": ["US"], "bailout": ["DE"]},
        {"cat": "1. US Shock", "shock": ["US"], "bailout": ["JP"]},
        {"cat": "1. US Shock", "shock": ["US"], "bailout": ["CA"]},

        # --- DE Shock ---
        {"cat": "2. AU Shock", "shock": ["AU"], "bailout": ["IT"]},
        {"cat": "2. DE Shock", "shock": ["DE"], "bailout": ["US"]},
        {"cat": "2. DE Shock", "shock": ["DE"], "bailout": ["GB"]},
        {"cat": "2. DE Shock", "shock": ["DE"], "bailout": ["JP"]},
        {"cat": "2. DE Shock", "shock": ["DE"], "bailout": ["CA"]},

        # --- LU Shock (Offshore Hub) ---
        {"cat": "3. LU Shock", "shock": ["LU"], "bailout": ["LU"]},
        {"cat": "3. LU Shock", "shock": ["LU"], "bailout": ["US"]},
        {"cat": "3. LU Shock", "shock": ["LU"], "bailout": ["GB"]},
        {"cat": "3. LU Shock", "shock": ["LU"], "bailout": ["DE"]},
        {"cat": "3. LU Shock", "shock": ["LU"], "bailout": ["JP"]},
        {"cat": "3. LU Shock", "shock": ["LU"], "bailout": ["CA"]},

        # --- JP Shock ---
        {"cat": "4. JP Shock", "shock": ["JP"], "bailout": ["JP"]},
        {"cat": "4. JP Shock", "shock": ["JP"], "bailout": ["US"]},
        {"cat": "4. JP Shock", "shock": ["JP"], "bailout": ["GB"]},
        {"cat": "4. JP Shock", "shock": ["JP"], "bailout": ["DE"]},
        {"cat": "4. JP Shock", "shock": ["JP"], "bailout": ["CA"]},

        # --- Extreme Case 1: Multiple Hub Failure ---
        {"cat": "5. Extreme: GB + HK", "shock": ["GB", "HK"], "bailout": ["GB", "HK"]}, # Direct Save
        {"cat": "5. Extreme: GB + HK", "shock": ["GB", "HK"], "bailout": ["US"]},
        {"cat": "5. Extreme: GB + HK", "shock": ["GB", "HK"], "bailout": ["DE", "JP"]}, # Firewall attempt

        # --- Extreme Case 2: Transatlantic Meltdown ---
        {"cat": "5. Extreme: US + GB", "shock": ["US", "GB"], "bailout": ["US", "GB"]},
        {"cat": "5. Extreme: US + GB", "shock": ["US", "GB"], "bailout": ["DE", "FR"]}, # EU Shield
        {"cat": "5. Extreme: US + GB", "shock": ["US", "GB"], "bailout": ["JP", "HK"]}, # Asian Shield

        # --- Extreme Case 3: EU Periphery Sovereign Crisis ---
        {"cat": "5. Extreme: IT + ES", "shock": ["IT", "ES"], "bailout": ["IT", "ES"]}, # Bailout the periphery
        {"cat": "5. Extreme: IT + ES", "shock": ["IT", "ES"], "bailout": ["DE", "FR"]}, # Let periphery burn, shield core
    ]
    
    print("\n" + "="*90)
    print(f"EXPERIMENT RESULTS: BAILOUT EFFICACY UNDER CREDIT CRUNCH ({int(CRUNCH_RATIO*100)}%)")
    print("="*90)
    
    results_data = []

    for exp in experiments:
        shock_codes = exp["shock"]
        bailout_codes = exp["bailout"]
        
        # Convert codes to indices
        try:
            shock_indices = [node_codes.index(c) for c in shock_codes]
            bailout_indices = [node_codes.index(c) for c in bailout_codes]
        except ValueError as e:
            print(f"Skipping {exp['cat']} - {e}")
            continue
            
        # Run Control (No Bailout)
        S_A, Assets_A = simulate_contagion_full_scenario(
            A_layers, shock_indices, b_vec, c_vec, EXTERNAL_ASSET_RATIO, 
            crunch_ratio=CRUNCH_RATIO, 
            bailout_indices=None
        )
        
        # Run Intervention (With Bailout)
        S_B, Assets_B = simulate_contagion_full_scenario(
            A_layers, shock_indices, b_vec, c_vec, EXTERNAL_ASSET_RATIO, 
            crunch_ratio=CRUNCH_RATIO, 
            bailout_indices=bailout_indices
        )
        
        # Calculate Metrics
        dead_A = np.sum(S_A[:, 0] == 1)
        dead_B = np.sum(S_B[:, 0] == 1)
        
        loss_A = np.sum(Assets_A[S_A[:, 0] == 1])
        loss_B = np.sum(Assets_B[S_B[:, 0] == 1])
        total_sys = np.sum(Assets_A)
        
        pct_A = (loss_A / total_sys) * 100
        pct_B = (loss_B / total_sys) * 100
        
        net_saved_count = dead_A - dead_B
        net_saved_asset_pct = pct_A - pct_B
        
        results_data.append({
            "Category": exp["cat"],
            "Shock Origin": "+".join(shock_codes),
            "Bailout Target": "+".join(bailout_codes),
            "Defaults (No Act)": dead_A,
            "Defaults (Saved)": dead_B,
            "Net Nodes Saved": net_saved_count,
            "Loss % (No Act)": f"{pct_A:.1f}%",
            "Loss % (Saved)": f"{pct_B:.1f}%",
            "Asset Preserved": f"{net_saved_asset_pct:.1f}%"
        })
        
    # Print Markdown Table
    df_res = pd.DataFrame(results_data)
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()