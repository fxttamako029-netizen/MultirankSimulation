# How to apply this cleanup patch

1. Download and unzip the patch.
2. Copy these files into the **root** of your local `MultirankSimulation` repository:
   - `README.md`
   - `requirements.txt`
   - `.gitignore`
   - `CITATION.cff`
   - `apply_cleanup.py`
3. From the repository root, run:

```bash
python3 apply_cleanup.py
```

The script will:

- replace the hard-coded `/Users/tamako029/...` data path with a portable repository-relative `data/` path;
- rename the scripts to ordered, descriptive filenames;
- run a syntax check without executing the simulations.

Then remove the one-time helper and commit the changes:

```bash
rm apply_cleanup.py
git status
git add -A
git commit -m "Improve reproducibility and repository documentation"
git push
```

On Windows PowerShell, remove the helper with:

```powershell
Remove-Item apply_cleanup.py
```

## Rename map

| Old filename | New filename |
| --- | --- |
| `bis_build_graph.py` | `01_build_network.py` |
| `bis_M2_plots.py` | `02_descriptive_plots.py` |
| `bis_M3_multirank.py` | `03_multirank_analysis.py` |
| `bis_M4_simulation.py` | `04_contagion_baseline.py` |
| `bis_M4v1_simulation.py` | `05_contagion_external_assets.py` |
| `bis_M4v2_save_market.py` | `06_bailout_experiment.py` |
| `bis_M4v3_hetrogeneous_buffer.py` | `07_heterogeneous_buffers.py` |
| `bis_M4v4_credit_crunch.py` | `08_credit_crunch.py` |
| `bis_M4v5_save_market(creditCrunch).py` | `09_bailout_credit_crunch_experiments.py` |
| `graph_decoration.py` | `10_multiplex_3d_visualization.py` |

`HOW_TO_APPLY.md` is only an instruction file for you; it does not need to be committed to the repository.
