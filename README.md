# VOI LabRanker

Concise, single-script pipeline to rank ICU lab tests by value-of-information.

## Script
- `lab_test_ranker.py`: downloads PhysioNet/CinC 2012 set-A, preprocesses 48h sequences, trains a dual-head Transformer (mortality + mech-vent within 6h), and produces paper-ready artifacts.

## Run
```bash
python3 lab_test_ranker.py
```
No flags required; configuration is in-script.

## Outputs
- `lab_rank_results/` with `summary.txt`, `tables/metrics.json`, `figs/`, and `best_model.pt`.

## Features
- Counterfactual masking utilities (cross-entropy gain per lab).
- Validation-only global scoring to avoid leakage.
- Budget and optional cost-aware evaluation.
- Redundancy audit and lead-time stats.
