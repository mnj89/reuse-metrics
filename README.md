# reuse-metrics: replication package

Code, data and results for the paper

> *Measuring Software Reuse When the Reusing Agent is a Foundation Model: A Provenance-Partition Framework* (under review).

The package contains the reference implementation of the framework (five provenance classes D, L, P, G, H; the
protocol tests T1-T4 and the model-swap test; the metrics DRR, LRD, PPC, PPA, GCI and HRS), and everything needed
to recompute the numbers, tables and figures of the paper.

## Quick start

```bash
git clone <this repository> && cd reuse-metrics
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install          # or: pip install -e ".[dev,studies]"
make all              # tests + field-study numbers + pilot numbers; about a minute, no network, no API keys
```

Expected: `37 passed`, a `results.json` identical to the stored one, and
`All pilot numbers reproduced from the stored answers.`

## Where each result of the paper comes from

| Paper | How to reproduce | Stored output |
|---|---|---|
| Section VII, Table II, Fig. 4 (worked example) | `make notebook`, or open `notebooks/worked_example.ipynb` | the executed notebook |
| Proposition 1; Theorems 3, 4, 6, 7 (numerical checks) | `make test` | `tests/test_metrics.py` |
| Section IX, Tables V-VII, RQ1-RQ3 (field study) | `make field-study` | `results/field_study/results.json`, `pooled_per_repo.csv` |
| Section IX, the four runs themselves | `studies/field_study/study_a_pipeline.py` (see `docs/REPRODUCE.md`) | `results/field_study/runs/` |
| Section IX-D, Table VIII (swap-test pilot) | `make pilot` | `results/swap_test_pilot/` |
| Figs. 1, 3, 4 (diagrams) and Figs. 2, 5 (data) | `make figures` | built into `figures/` |

`docs/REPRODUCE.md` gives the steps, run times and expected values in detail.

## Layout

```
src/reuse_metrics/        the library: model, partition (T1-T4, swap test), metrics, synthetic generator
tests/                    37 tests: the worked example and the formal results
examples/                 synthetic end-to-end demo with a planted ground truth
notebooks/                executed notebook of the worked example, and the script that builds it
studies/
  field_study/            pipeline over the AIDev dataset, the 54-term vocabulary, the analysis script
  swap_test_pilot/        block extraction, model clients, the swap-test runner, analysis, Colab notebook
results/
  field_study/            the four runs, the pooled table, every reported number
  swap_test_pilot/        the 30 blocks, all verdicts, every model answer
figures/                  sources of all five figures
docs/                     reproduction guide, guide to running the swap test, data notes
```

## Using the library

```python
from reuse_metrics import Attribution, Event, ProtocolParams, System, Unit, partition, report

units = [Unit("app.py", i, "r1", content=f"line {i}") for i in range(3)]
attrs = {units[0].key: Attribution("flask", "import")}
events = [Event("e1", 1.0, "add CRUD routes", "model-A", frozenset({units[1].key}),
                emitted_text={units[1].key: "line 1"}, block_of={units[1].key: "create"})]
system = System(units, attrs, events)

params = ProtocolParams(rho0=0.5, pattern_set={"SQLAlchemy"}, alt_models={"model-B"}, theta=0.3)
est = partition(system, params, oracle=None)     # oracle=None skips the swap test (no G verdicts)
print(report(est, events, params.pattern_set, reconstructed={"e1"}))
```

Every reported value should carry the parameter tuple from `ProtocolParams.as_dict()` and the unit granularity.

## Data and licences

- The field study uses the public AIDev dataset, release `v5` (https://huggingface.co/datasets/hao-li/AIDev). It is
  not redistributed here; the pipeline downloads it. See `docs/DATA.md`.
- `results/` holds derived data. The pilot files contain excerpts of code and pull-request text from public
  repositories, which remain under the licences of those repositories.
- The code is released under the MIT licence (`LICENSE`). To cite, see `CITATION.cff`.

## Limits

The field-study numbers are deterministic given the dataset release and the seeds. The swap-test pilot used hosted
models that may change or be withdrawn, so re-running it against the models will not return the same answers; the
stored answers make every reported pilot number recomputable. The pilot is exploratory and is not an estimate of GCI.
