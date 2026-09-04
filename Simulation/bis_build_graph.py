import os
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# -----------------------------
# 1) Your local file locations
# -----------------------------
DATA_DIR = r"/Users/tamako029/Desktop/Honours Thesis/code"

CSV_FILES = [
    "2006Q1-2007Q2.csv",
    "2007Q3-2008Q4.csv",
    "2009Q1-2009Q4.csv",
]

# Choose which slice to visualize (keep it fixed for comparability)
POSITION = "C:Total claims"         # or "L:Total liabilities"
INSTRUMENT = "A:All instruments"    # or "G:Loans and deposits"

# Optional: edge filtering for cleaner plots (useful for demo)
USE_EDGE_FILTER = True  # set True to enable filtering
TOP_K_OUT_EDGES = 10     # only used when USE_EDGE_FILTER is True


# -----------------------------
# 2) Read BIS export CSV safely
# -----------------------------
def read_bis_export(csv_path: str) -> pd.DataFrame:
    """
    BIS Data Portal CSV export often includes metadata lines above the true header.
    We find the header row by searching for the line that starts with:
    'DATAFLOW_ID:Dataflow ID'
    """
    lines = Path(csv_path).read_text(errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("DATAFLOW_ID:Dataflow ID"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot find BIS header row in: {csv_path}")

    return pd.read_csv(csv_path, skiprows=header_idx, encoding="utf-8-sig")


# -----------------------------
# 3) Build a graph for one quarter
# -----------------------------
def build_graph_for_quarter(df: pd.DataFrame, quarter: pd.Timestamp) -> nx.DiGraph:
    G = nx.DiGraph()

    sub = df[df["TIME_PERIOD:Period"] == quarter].copy()

    # Node set: union of reporting + counterparty countries in this quarter
    nodes = sorted(set(sub["L_REP_CTY:Reporting country"]).union(set(sub["L_CP_COUNTRY:Counterparty country"])))
    G.add_nodes_from(nodes)

    # Add directed weighted edges
    for _, r in sub.iterrows():
        u = r["L_REP_CTY:Reporting country"]
        v = r["L_CP_COUNTRY:Counterparty country"]
        if u == v:
            continue  # skip self-loop for readability

        w = r["OBS_VALUE:Value"]
        if pd.isna(w) or float(w) <= 0:
            continue  # skip missing or non-positive weights

        G.add_edge(u, v, weight=float(w))

    return G

def top_k_out_edges(G: nx.DiGraph, k: int) -> list[tuple[str, str]]:
    """Return an edge list keeping only the top-k outgoing edges per node by weight."""
    keep = set()
    for u in G.nodes():
        outs = [(u, v, G[u][v]["weight"]) for v in G.successors(u)]
        outs.sort(key=lambda x: x[2], reverse=True)
        for u2, v2, _ in outs[:k]:
            keep.add((u2, v2))
    return list(keep)


# -----------------------------
# 4) Main: one representative quarter per CSV
# -----------------------------
def main():

    rep_graphs = {}  # filename -> (quarter, graph)

    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        df = read_bis_export(path)

        # Parse quarter
        df["TIME_PERIOD:Period"] = pd.to_datetime(df["TIME_PERIOD:Period"], errors="coerce")

        # Filter to your chosen slice
        df = df[
            (df["L_POSITION:Balance sheet position"] == POSITION) &
            (df["L_INSTR:Type of instruments"] == INSTRUMENT)
        ].copy()

        # Choose the representative quarter = last quarter in this file
        quarters = sorted(df["TIME_PERIOD:Period"].dropna().unique())
        if not quarters:
            raise ValueError(f"No quarters found after filtering in file: {fname}")

        q_rep = quarters[-1]
        G_rep = build_graph_for_quarter(df, q_rep)
        rep_graphs[fname] = (q_rep, G_rep)

    # One consistent layout for all three graphs (easier to compare)
    G_union = nx.DiGraph()
    for _, (_, G) in rep_graphs.items():
        G_union.add_nodes_from(G.nodes())
        G_union.add_edges_from(G.edges())

    pos = nx.spring_layout(G_union, seed=42, k=1.2)
    # Use one fixed layout so node positions are comparable across the three segments

    # Draw and show each representative graph
    for fname, (q_rep, G) in rep_graphs.items():
        plt.figure(figsize=(14, 10))

        # Labels as full country names (e.g., "United States")
        labels = {n: (n.split(":", 1)[1] if ":" in n else n) for n in G.nodes()}

        nx.draw_networkx_nodes(G, pos, node_size=1400)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))

        # Choose whether to filter edges (for cleaner plots)
        if USE_EDGE_FILTER:
            edges_to_draw = top_k_out_edges(G, TOP_K_OUT_EDGES)
        else:
            edges_to_draw = list(G.edges())

        if edges_to_draw:
            weights = np.array([G[u][v]["weight"] for (u, v) in edges_to_draw], dtype=float)

            # Simple log scaling for edge widths
            widths = np.log10(weights + 1.0)
            if widths.max() > 0:
                widths = 0.5 + 4.0 * (widths / widths.max())

            nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, arrows=True, arrowsize=18, width=widths, alpha=0.7)

        plt.title(f"{POSITION} | {INSTRUMENT}\nRepresentative quarter (last in file): {q_rep.date()}\nSource: {fname}")
        plt.axis("off")
        plt.tight_layout()
        plt.margins(0.15)

        # Show the figure in a window instead of saving
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()