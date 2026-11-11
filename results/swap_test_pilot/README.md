# Swap-test feasibility pilot (Section IX-D, Table VIII)

30 whole functions with a docstring added by merged Claude Code pull requests, run in November 2026 with
`anthropic:claude-haiku-5-5`, `anthropic:claude-sonnet-5-5` (also the proxy for the originating model) and
`ollama:gpt-oss:120b`; R = 3, q = 0.34, theta in {0.3, 0.4, 0.5}. The Claude models ran at their default temperature.

| Folder | Prompt | Distance | Generation calls |
|---|---|---|---|
| `pr` | PR title and body | token | 270 |
| `doc` | PR text + the function's docstring | token | 270 |
| `pr_alpha` | as `pr` | local names renamed | 0 (re-scored from `pr/cache.jsonl`) |
| `doc_alpha` | as `doc` | local names renamed | 0 (re-scored from `doc/cache.jsonl`) |

Recompute all four from the stored answers with `make pilot`. File formats are described in `docs/DATA.md`.
The pilot is exploratory and is not an estimate of GCI.
