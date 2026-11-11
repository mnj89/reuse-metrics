# Reproduction guide

All commands run from the repository root after `make install`. Levels 1-3 need no network and no API keys.

## Level 1: library and worked example (seconds)

```bash
make test        # 37 passed
make example     # synthetic system: per-unit agreement with the planted truth is 1.000
make notebook    # re-executes notebooks/worked_example.ipynb
```

The notebook prints the values of Section VII: DRR 0.15, LRD 0.40, PPC 0.20, PPA 0.80, GCI 0.20, HRS 0.475.

## Level 2: field study from the stored runs (seconds)

```bash
make field-study
```

Reads `results/field_study/runs/*/per_repo_metrics.csv`, pools the three same-parameter runs, and rewrites
`results/field_study/results.json` and `pooled_per_repo.csv`. `git diff results/` should then show no change.

| Quantity (paper) | Key in `results.json` | Value |
|---|---|---|
| Repositories, pull requests, units | `pooled_n_repos`, `pooled_n_events`, `pooled_n_units` | 105; 1,528; 1,095,235 |
| Medians (Table V) | `pooled_stats.median` | DRR 0.036, LRD 0.887, PPC 0.111, PPC_U 0.054, HRS 0.456 |
| Equal-DRR pairs (RQ2) | `equal_drr_pairs_total`, `..._lrd_gt_0.1`, `..._lrd_gt_0.3`, `..._ppc_gt_0.2` | 1,536; 874; 363; 344 |
| Largest LRD gap at equal DRR | `max_lrd_gap_at_equal_drr` | 0.92 |
| Medians by run (Table VII) | `per_run_median` | see file |
| Per-agent medians (Table VI) | `per_agent_median` | see file |

## Level 3: swap-test pilot from the stored answers (seconds)

```bash
make pilot
```

Re-scores the 30 blocks from `results/swap_test_pilot/{pr,doc}/cache.jsonl` for the four configurations of
Table VIII and compares every count with the stored `summary.json`. No model is called.

## Level 4: field study from the raw dataset (about 30 minutes per run, 12 GB download)

```bash
python studies/field_study/study_a_pipeline.py --scope pop --n-repos 40 --seed 1 --out output_seed1_min5
python studies/field_study/study_a_pipeline.py --scope pop --n-repos 40 --seed 2 --out output_seed2_min5
python studies/field_study/study_a_pipeline.py --scope pop --n-repos 40 --seed 3 --out output_seed3_min5
python studies/field_study/study_a_pipeline.py --scope pop --n-repos 40 --seed 1 --min-prs 10 --out output_seed1_min10
```

Each run writes `summary.json`, `per_repo_metrics.csv`, `equal_drr_pairs.csv`, `table_rq1.tex` and `units.parquet`;
compare them with `results/field_study/runs/`. The exact command and parameters of each stored run are recorded in its
`summary.json`; use those if they differ from the lines above. The dataset revision is pinned to `v5`.
To exercise the pipeline without the download:

```bash
python studies/field_study/make_fake_aidev.py fake_aidev
python studies/field_study/study_a_pipeline.py --data-dir fake_aidev --n-repos 10 --out output_fake   # meaningless numbers
```

## Level 5: swap test against live models (API usage; answers will differ)

See `docs/RUNNING_THE_SWAP_TEST.md`.

## Figures

`make figures` needs `pdflatex` with TikZ, `lmodern` and `sansmath`. Figures 1, 3 and 4 are TikZ sources; Figures 2
and 5 are drawn by `figures/figs_data.py` (Figure 5 from `results/field_study/pooled_per_repo.csv`; Figure 2 from
values quoted from the two cited public reports).
