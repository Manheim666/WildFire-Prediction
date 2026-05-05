#!/usr/bin/env python3
"""
MANHEIM Wildfire Prediction — Full Pipeline Runner
===================================================
Executes all 6 notebooks sequentially: NB1 → NB2 → NB3 → NB4 → NB5 → NB6.
Each notebook is run in-place using nbclient with per-cell tqdm progress.
If any notebook fails, the pipeline stops immediately and reports the error.

Usage:
    python run_pipeline.py                  # run all 6 notebooks (no timeout)
    python run_pipeline.py --from 4         # start from NB04
    python run_pipeline.py --only 5         # run only NB05
    python run_pipeline.py --timeout 7200   # 2 h per-cell timeout
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from tqdm import tqdm

# ── Notebook order ────────────────────────────────────────────────────────
NOTEBOOKS = [
    "01_Data_Ingestion.ipynb",
    "02_EDA_FeatureEngineering.ipynb",
    "03_Weather_TimeSeries.ipynb",
    "04_Wildfire_Detection.ipynb",
    "05_Risk_Prediction_Dashboard.ipynb",
    "06_Climate_Report.ipynb",
]

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
LOG_DIR = PROJECT_ROOT / "logs"

# Estimated runtimes (minutes) for progress display
ESTIMATED_MINUTES = [15, 5, 120, 20, 5, 2]


def run_notebook(nb_path: Path, timeout: int, kernel: str = "python3") -> tuple[bool, float, str]:
    """Execute a notebook in-place with per-cell tqdm progress.

    Returns (success, elapsed_seconds, error_msg).
    """
    nb = nbformat.read(nb_path, as_version=4)
    cell_timeout = timeout if timeout > 0 else None  # None = unlimited

    client = NotebookClient(
        nb,
        timeout=cell_timeout,
        kernel_name=kernel,
    )

    code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    short_name = nb_path.stem

    t0 = time.time()
    try:
        with client.setup_kernel(cwd=str(nb_path.parent)):
            pbar = tqdm(
                code_cells,
                desc=f"  {short_name}",
                unit="cell",
                leave=True,
                bar_format="  {desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} cells [{elapsed}<{remaining}]",
            )
            for cell_index, cell in pbar:
                client.execute_cell(cell, cell_index)
            pbar.close()
    except Exception as exc:
        elapsed = time.time() - t0
        # Extract short error message
        tb_lines = traceback.format_exception_only(type(exc), exc)
        short_err = tb_lines[-1].strip() if tb_lines else str(exc)
        return False, elapsed, short_err

    elapsed = time.time() - t0
    # Save executed notebook in-place
    nbformat.write(nb, nb_path)
    return True, elapsed, ""


def fmt_time(seconds: float) -> str:
    """Format seconds as 'Xm Ys'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(
        description="MANHEIM Pipeline — run all notebooks sequentially"
    )
    parser.add_argument(
        "--from", dest="start_from", type=int, default=1,
        help="Start from notebook N (1-6). Default: 1"
    )
    parser.add_argument(
        "--only", type=int, default=0,
        help="Run only notebook N (1-6)"
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="Per-cell timeout in seconds. 0 = no timeout (default). e.g. --timeout 7200 for 2h"
    )
    parser.add_argument(
        "--kernel", type=str, default="python3",
        help="Jupyter kernel name. Default: python3"
    )
    args = parser.parse_args()

    # Determine which notebooks to run
    if args.only > 0:
        indices = [args.only - 1]
    else:
        indices = list(range(args.start_from - 1, len(NOTEBOOKS)))

    selected = [(i, NOTEBOOKS[i]) for i in indices if 0 <= i < len(NOTEBOOKS)]
    if not selected:
        print("No valid notebooks selected.")
        sys.exit(1)

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Header
    total = len(selected)
    print("=" * 64)
    print("  MANHEIM Wildfire Prediction — Pipeline Runner")
    print("=" * 64)
    print(f"  Notebooks  : {total} ({', '.join(nb for _, nb in selected)})")
    print(f"  Timeout    : {args.timeout}s per cell" if args.timeout > 0 else "  Timeout    : unlimited")
    print(f"  Kernel     : {args.kernel}")
    print(f"  Project    : {PROJECT_ROOT}")
    print("=" * 64)
    print()

    pipeline_t0 = time.time()
    results = []

    for step, (idx, nb_name) in enumerate(selected, 1):
        nb_path = NOTEBOOK_DIR / nb_name
        if not nb_path.exists():
            print(f"[{step}/{total}] ✗ {nb_name} — FILE NOT FOUND")
            results.append((nb_name, False, 0, "File not found"))
            break

        est = ESTIMATED_MINUTES[idx]
        print(f"[{step}/{total}] Running {nb_name}  (est. ~{est} min) …")

        success, elapsed, error = run_notebook(nb_path, args.timeout, args.kernel)

        if success:
            print(f"         ✓ Completed in {fmt_time(elapsed)}")
            results.append((nb_name, True, elapsed, ""))
        else:
            print(f"         ✗ FAILED after {fmt_time(elapsed)}")
            print(f"           Error: {error[:200]}")
            results.append((nb_name, False, elapsed, error))
            print(f"\n  Pipeline stopped at {nb_name}.")
            break

        print()

    # Summary
    pipeline_elapsed = time.time() - pipeline_t0
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    n_fail = sum(1 for _, ok, _, _ in results if not ok)

    print()
    print("=" * 64)
    print("  PIPELINE SUMMARY")
    print("=" * 64)
    for nb_name, ok, elapsed, error in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {nb_name:45s} {fmt_time(elapsed):>8s}")
        if not ok and error:
            print(f"    Error: {error[:120]}")
    print("-" * 64)
    print(f"  Total: {n_ok} passed, {n_fail} failed — {fmt_time(pipeline_elapsed)}")
    print("=" * 64)

    # Write log
    log_path = LOG_DIR / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w") as f:
        f.write(f"MANHEIM Pipeline Run — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*64}\n")
        for nb_name, ok, elapsed, error in results:
            status = "OK" if ok else "FAIL"
            f.write(f"{status:4s}  {nb_name:45s}  {fmt_time(elapsed):>8s}\n")
            if not ok and error:
                f.write(f"      {error}\n")
        f.write(f"{'='*64}\n")
        f.write(f"Total: {n_ok} passed, {n_fail} failed — {fmt_time(pipeline_elapsed)}\n")
    print(f"\n  Log: {log_path}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
