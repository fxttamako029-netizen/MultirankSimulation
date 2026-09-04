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
SHOCK_ORIGIN = "US"      # The Crisis Origin (e.g., Lehman Brothers)
BAILOUT_TARGET = "GB"    # The Intervention Target (e.g., UK Bailout)

# --- Thesis Simulation Parameters (High Stress) ---
# We use stricter buffers to ensure contagion actually happens, 
# making the bailout visible.
BUFFER_LAYER = 0.03   # 3% (Interbank Freeze threshold)
BUFFER_TOTAL = 0.05   # 5% (Insolvency threshold)
EXTERNAL_ASSET_RATIO = 0.20

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
# 2. MODIFIED SIMULATION (With Intervention Logic)
# =========================================================
def simulate_contagion(
    A_layers, 
    initial_victim_idx, 
    b_buffer, 
    c_buffer, 
    external_asset_ratio=0.20, 
    max_steps=100,
    bailout_idx=None  # <--- NEW ARGUMENT: The index of the country to save
):
    L = len(A_layers)
    N = A_layers[0].shape[0]
    
    # 1. Asset Estimation
    Assets_network = np.zeros((N, L))
    for alpha in range(L):
        Assets_network[:, alpha] = np.sum(A_layers[alpha], axis=1)
    
    Assets_total_estimated = np.sum(Assets_network, axis=1) / external_asset_ratio
    Assets_total_estimated[Assets_total_estimated == 0] = 1.0 
    Assets_layer_strict = Assets_network
    Assets_layer_strict[Assets_layer_strict == 0] = 1.0
    
    # 2. Initialize State
    S = np.zeros((N, L), dtype=int) 
    
    if initial_victim_idx is not None:
        S[initial_victim_idx, :] = 1 
    
    # 3. Contagion Loop
    for t in range(max_steps):
        S_prev = S.copy()
        
        Loss_layer = np.zeros((N, L))
        for alpha in range(L):
            Loss_layer[:, alpha] = A_layers[alpha] @ S[:, alpha]
            
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_layer = Loss_layer / Assets_layer_strict 
            ratio_layer[np.isnan(ratio_layer)] = 0
            
        condition_horizontal = ratio_layer > b_buffer
        
        Total_Loss = np.sum(Loss_layer, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_total = Total_Loss / Assets_total_estimated
            ratio_total[np.isnan(ratio_total)] = 0
            
        condition_vertical = ratio_total > c_buffer
        
        # Apply Failures
        S[condition_horizontal] = 1 
        for i in range(N):
            if condition_vertical[i]:
                S[i, :] = 1 
        
        # --- THE INTERVENTION (Circuit Breaker) ---
        # If a bailout target is defined, we force it to stay healthy (0)
        # even if the math above says it should fail.
        if bailout_idx is not None:
            # Check if it *would* have failed (for reporting purposes, optional)
            # Force State to 0
            S[bailout_idx, :] = 0
            
        if np.array_equal(S, S_prev):
            break
            
    return S

# =========================================================
# 3. VISUALIZATION
# =========================================================
def plot_comparison_bar(df_res, title):
    """Visualizes the number of defaults in two scenarios."""
    plt.figure(figsize=(8, 5))
    colors = ['#d62728', '#2ca02c'] # Red for No Bailout, Green for Bailout
    plt.bar(df_res['Scenario'], df_res['Defaults'], color=colors, edgecolor='black')
    plt.ylabel('Number of Defaulted Countries')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

# =========================================================
# 4. MAIN EXECUTION (Comparative Experiment)
# =========================================================
def main():
    # Load
    frames = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"File not found: {path} (Skipping)")
            continue
        df = read_bis_export(path)
        if df.empty: continue
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")
        frames.append(df)
    
    if not frames:
        print("No valid data loaded.")
        return
    df_all = pd.concat(frames, ignore_index=True)

    # Build Network
    node_labels, node_codes, node_names = build_nodes(df_all)
    A_layers = build_A_layers_for_quarter(df_all, TARGET_QUARTER, node_labels)
    N = len(node_labels)
    
    print(f"--- Intervention Simulation: {TARGET_QUARTER.date()} ---")
    print(f"Shock Origin: {SHOCK_ORIGIN}")
    print(f"Bailout Target: {BAILOUT_TARGET}")
    
    # Get Indices
    try:
        shock_idx = node_codes.index(SHOCK_ORIGIN)
        bailout_idx = node_codes.index(BAILOUT_TARGET)
    except ValueError:
        print("Country code not found.")
        return

    # --- SCENARIO A: NO BAILOUT ---
    print("\nRunning Scenario A: No Intervention...")
    S_A = simulate_contagion(
        A_layers, shock_idx, BUFFER_LAYER, BUFFER_TOTAL, EXTERNAL_ASSET_RATIO,
        bailout_idx=None # <--- No one is saved
    )
    defaults_A = np.sum(S_A[:, 0] == 1)
    
    # --- SCENARIO B: WITH BAILOUT ---
    print(f"Running Scenario B: Bailing out {BAILOUT_TARGET}...")
    S_B = simulate_contagion(
        A_layers, shock_idx, BUFFER_LAYER, BUFFER_TOTAL, EXTERNAL_ASSET_RATIO,
        bailout_idx=bailout_idx # <--- Intervention Active
    )
    defaults_B = np.sum(S_B[:, 0] == 1)
    
    # --- RESULTS ANALYSIS ---
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Total Defaults (No Bailout):   {defaults_A}")
    print(f"Total Defaults (With Bailout): {defaults_B}")
    print(f"Net Lives Saved:               {defaults_A - defaults_B}")
    
    # Identify who was saved
    saved_indices = [i for i in range(N) if S_A[i, 0] == 1 and S_B[i, 0] == 0]
    
    if saved_indices:
        print("\nCountries Saved by the Intervention:")
        saved_data = []
        for i in saved_indices:
            # Calculate exposure to the Bailout Target (Why were they saved?)
            exposure_to_target = A_layers[0][i, bailout_idx]
            saved_data.append({
                "Code": node_codes[i],
                "Name": node_names[i],
                "Direct Exposure to Target": exposure_to_target
            })
        
        df_saved = pd.DataFrame(saved_data).sort_values("Direct Exposure to Target", ascending=False)
        print(df_saved.to_string(index=False))
        
        # Plot
        df_plot = pd.DataFrame({
            'Scenario': ['No Bailout', f'Bailout {BAILOUT_TARGET}'],
            'Defaults': [defaults_A, defaults_B]
        })
        plot_comparison_bar(df_plot, f"Impact of Bailing Out {BAILOUT_TARGET}\n(Shock Origin: {SHOCK_ORIGIN})")
    else:
        print("\nNo additional countries were saved.")
        print("Possible reasons: 1) Target didn't fail in Scenario A. 2) Target wasn't a key contagion hub.")

if __name__ == "__main__":
    main()