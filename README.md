# Atom: Text-Guided Geometric Relation Parsing

This repository contains the paper-facing code for a lightweight geometric relation parser built on a derived Ext-PGDP5K protocol.

The public package is intentionally clean:

- no private logs
- no training outputs
- no full PGDP5K data dump
- no remote-server credentials or deployment helpers

Instead, it includes a tiny runnable demo subset so the project can be cloned and executed immediately.

## Repository Layout

- `src/`: model, data loader, logic regularization, metrics
- `train.py`: training entry point
- `scripts/`: protocol and evaluation utilities
- `docs/ext_pgdp5k_protocol.md`: derived protocol notes
- `demo_data/PGDP5K_demo/`: small metadata-only demo split
- `run_demo.ps1`: one-click Windows demo
- `run_demo.bat`: one-click Windows batch wrapper

## Quick Start

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo:

```powershell
.\run_demo.ps1
```

Or:

```bat
run_demo.bat
```

The script trains for one short epoch on the included demo subset and writes outputs to `outputs/demo_run`.

## Demo Data

The included `demo_data/PGDP5K_demo` subset is a tiny metadata-only package prepared from a few samples for reproducibility checks. It is not intended for reporting final paper numbers.

The full PGDP5K dataset and the full derived Ext-PGDP5K protocol should be prepared separately for real experiments.

## Example Manual Command

```bash
python train.py ^
  --data-root demo_data/PGDP5K_demo ^
  --ext-root demo_data/PGDP5K_demo/Ext-PGDP5K ^
  --epochs 1 ^
  --batch-size 2 ^
  --device cpu ^
  --output-dir outputs/demo_run
```

## Notes

- The demo package is designed for smoke testing and code review.
- Final paper experiments should be run on the full dataset.
- The current active relation setting contains four evaluated labels:
  `intersect`, `parallel`, `perpendicular`, and `bisect`.
