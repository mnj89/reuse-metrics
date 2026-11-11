# Running the model-swap test on new blocks

The pilot of the paper can be re-scored without any model access (`make pilot`). This guide is for running the
test on new blocks or with other models. It costs API usage and its answers will differ from the stored ones.

## Providers

`llm_oracle.py` accepts model specs of the form `provider:model-id`.

| Provider | Needs | Notes |
|---|---|---|
| `anthropic:` | `ANTHROPIC_API_KEY` | Models that reject a custom temperature run at their default; this is recorded in `summary.json` |
| `openai:` | `OPENAI_API_KEY` | |
| `google:` | `GOOGLE_API_KEY` | |
| `glm:` | `GLM_API_KEY`, optional `GLM_BASE_URL` | Any OpenAI-compatible endpoint |
| `ollama:` | `OLLAMA_API_KEY` for Ollama Cloud, or a local Ollama server | Runs at the requested temperature and seed |
| `claudecli:` | the `claude` command, signed in | Calls `claude -p` with no tools; temperature cannot be set |
| `fake:` | nothing | Deterministic stub for plumbing tests |

Keys are read from the environment. Never write a key into a file or a notebook cell.

## Steps (from the repository root)

```bash
pip install -e ".[studies,models]"

# 1. check keys and model IDs with one short call each
python studies/swap_test_pilot/smoke_test_keys.py anthropic:<small> anthropic:<large> ollama:<open-model>

# 2. extract whole functions added by agent pull requests (downloads AIDev v5, about 12 GB)
python studies/swap_test_pilot/extract_blocks.py --repos results/field_study/pooled_per_repo.csv \
    --n-blocks 30 --only-agents Claude_Code --require-docstring --seed 2 --out gci_new/blocks.jsonl

# 3. run the protocol (answers are cached: safe to interrupt and re-run)
python studies/swap_test_pilot/gci_run.py --blocks gci_new/blocks.jsonl \
    --models anthropic:<small> anthropic:<large> ollama:<open-model> \
    --origin Claude_Code=anthropic:<large> \
    --theta 0.3 0.4 0.5 --R 3 --q 0.34 --prompt-level doc --distance token \
    --per-repo results/field_study/pooled_per_repo.csv --out gci_new/doc

# 4. re-score the same answers with another distance (no API calls)
python studies/swap_test_pilot/gci_run.py ... --distance alpha --cache gci_new/doc/cache.jsonl --out gci_new/doc_alpha

# 5. tables, figures and a draft subsection for a full-size run
python studies/swap_test_pilot/gci_analyse.py --run gci_new/doc --models <same list> \
    --vendors anthropic=Anthropic ollama:<open-model>=<maker> --theta-main 0.4
```

`GCI_study_colab.ipynb` runs the same steps on Google Colab.

## Reading the output

`summary.json` gives, per threshold, the verdict counts (`under_specified`, `coupled`, `regenerable`; `*_no_control`
for agents whose originating vendor is not in the model set), the coupled share for nested model sets (the ordering of
Theorem 3), and F(delta) of Theorem 5. Only blocks that pass the self-regeneration control can be judged coupled.
The originating model version is not recorded in AIDev, so `--origin` names a proxy; say so when reporting.
