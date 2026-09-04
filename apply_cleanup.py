#!/usr/bin/env python3
"""One-time cleanup utility for the MultirankSimulation repository.

Run this file from the repository root after copying the cleanup patch files
into the repository. It only changes filenames and the machine-specific data
path; it does not alter model formulas or simulation parameters.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


RENAMES = {
    "bis_build_graph.py": "01_build_network.py",
    "bis_M2_plots.py": "02_descriptive_plots.py",
    "bis_M3_multirank.py": "03_multirank_analysis.py",
    "bis_M4_simulation.py": "04_contagion_baseline.py",
    "bis_M4v1_simulation.py": "05_contagion_external_assets.py",
    "bis_M4v2_save_market.py": "06_bailout_experiment.py",
    "bis_M4v3_hetrogeneous_buffer.py": "07_heterogeneous_buffers.py",
    "bis_M4v4_credit_crunch.py": "08_credit_crunch.py",
    "bis_M4v5_save_market(creditCrunch).py": "09_bailout_credit_crunch_experiments.py",
    "graph_decoration.py": "10_multiplex_3d_visualization.py",
}

PORTABLE_DATA_LINE = (
    'DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")'
    '  # repository-relative data directory'
)

DATA_DIR_PATTERN = re.compile(
    r'^\s*DATA_DIR\s*=\s*r?["\'][^"\']*["\']\s*(?:#.*)?$',
    flags=re.MULTILINE,
)


def detect_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), here]
    for root in candidates:
        if (root / "Simulation").is_dir() and (root / "data").is_dir():
            return root
    raise SystemExit(
        "Could not find the repository root. Put this file in the "
        "MultirankSimulation root and run: python apply_cleanup.py"
    )


def fix_data_paths(sim_dir: Path) -> tuple[int, list[str]]:
    changed = 0
    warnings: list[str] = []

    for path in sorted(sim_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "DATA_DIR" not in text:
            continue

        new_text, count = DATA_DIR_PATTERN.subn(PORTABLE_DATA_LINE, text, count=1)
        if count == 0:
            warnings.append(f"Could not rewrite DATA_DIR in {path.name}")
            continue

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1

    return changed, warnings


def rename_scripts(sim_dir: Path) -> tuple[int, list[str]]:
    renamed = 0
    warnings: list[str] = []

    for old_name, new_name in RENAMES.items():
        src = sim_dir / old_name
        dst = sim_dir / new_name

        if dst.exists() and not src.exists():
            continue  # already renamed
        if dst.exists() and src.exists():
            warnings.append(
                f"Skipped {old_name}: destination {new_name} already exists."
            )
            continue
        if not src.exists():
            warnings.append(f"Source file not found: {old_name}")
            continue

        src.rename(dst)
        renamed += 1

    return renamed, warnings


def syntax_check(sim_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(sim_dir.glob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.name}: {exc.msg} (line {exc.lineno})")
    return errors


def main() -> int:
    root = detect_root()
    sim_dir = root / "Simulation"

    print(f"Repository: {root}")
    print("1) Rewriting machine-specific DATA_DIR values...")
    path_changes, path_warnings = fix_data_paths(sim_dir)
    print(f"   Updated {path_changes} script(s).")

    print("2) Renaming research scripts to descriptive ordered filenames...")
    rename_count, rename_warnings = rename_scripts(sim_dir)
    print(f"   Renamed {rename_count} script(s).")

    print("3) Running a Python syntax check (no simulations are executed)...")
    syntax_errors = syntax_check(sim_dir)

    warnings = path_warnings + rename_warnings
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f" - {warning}")

    if syntax_errors:
        print("\nSyntax check found errors:")
        for error in syntax_errors:
            print(f" - {error}")
        print("\nThe cleanup completed, but review these files before committing.")
        return 1

    print("\nCleanup complete. Model formulas and experiment parameters were not changed.")
    print("\nSuggested next commands:")
    print("  rm apply_cleanup.py")
    print("  git status")
    print("  git add -A")
    print('  git commit -m "Improve reproducibility and repository documentation"')
    print("  git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
