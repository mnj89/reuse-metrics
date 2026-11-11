# Data notes

## Third-party data (not redistributed)

**AIDev**, release `v5` (cutoff 31 March 2026): https://huggingface.co/datasets/hao-li/AIDev.
Tables used: `pull_request` (the AIDev-pop subset, repositories with more than 100 stars) and
`pr_commit_details` (per-file diffs, about 12 GB). The pipeline streams the diffs and keeps Python changes only.

## Derived data in this repository

### `results/field_study/runs/<run>/`

| File | Content |
|---|---|
| `summary.json` | medians, interquartile ranges, parameters, hash of the vocabulary file |
| `per_repo_metrics.csv` | one row per repository: agent, counts per class (`n_D`, `n_L`, `n_P`, ...), DRR, LRD_log, PPC, PPC_U, PLC, HRS |
| `equal_drr_pairs.csv` | repository pairs with equal DRR and different profiles |
| `units.parquet` | every unit (added logical line) with its class |
| `table_rq1.tex` | the run's summary row as LaTeX |

Runs: `seed1_min5`, `seed2_min5`, `seed3_min5` (pooled in the paper) and `seed1_min10` (stricter inclusion threshold).
`results/field_study/pooled_per_repo.csv` is the pooled table of 105 repositories; `results.json` holds every number
reported in Section IX.

### `results/swap_test_pilot/`

| Path | Content |
|---|---|
| `blocks/blocks.jsonl` | the 30 functions: repository, pull request, file, signature, code, docstring, PR text |
| `<config>/verdicts.csv` | per block: distance under each model, self-regeneration distance, verdict at each threshold |
| `<config>/summary.json` | counts per threshold, nested-model-set shares, parameters |
| `<config>/per_repo_gci.csv` | per-repository coupled line share |
| `pr/cache.jsonl`, `doc/cache.jsonl` | every model answer (key, model, sample index, text) |

Configurations: `pr`, `doc` (prompt), each with the token distance and, as `*_alpha`, with local names renamed.
`doc/cache.jsonl` holds 54 extra answers (sample indices 3 and 4) from an interrupted five-sample run; they are not used.

## Vocabulary

`studies/field_study/patterns.txt` is the 54-term named-pattern vocabulary N. Its SHA-256 prefix is recorded in each
run's `summary.json`.
