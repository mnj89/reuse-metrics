# Studies

| Folder | Paper | Entry points |
|---|---|---|
| `field_study/` | Section IX (RQ1-RQ3) | `study_a_pipeline.py` (dataset to per-repository metrics), `analyse.py` (pooled numbers), `patterns.txt` (vocabulary N), `make_fake_aidev.py` (offline test data) |
| `swap_test_pilot/` | Section IX-D | `extract_blocks.py`, `gci_run.py` (protocol), `llm_oracle.py` (model clients, distance, cache), `gci_analyse.py` (tables and figures for a full-size run), `rescore_pilot.py` (recompute Table VIII), `smoke_test_keys.py`, `GCI_study_colab.ipynb` |

See `docs/REPRODUCE.md` for commands and `docs/RUNNING_THE_SWAP_TEST.md` for running the swap test against models.

What the field-study pipeline does not do: it does not run the swap test (no unit is classed G), it does not observe
post-merge human edits, and it has no human-authored comparison set unless `--human-prs` and `--human-commits` are given.
