"""Run the model-swap regeneration test (Section VIII-C) on extracted blocks and
compute GCI, the Theorem 3 monotonicity check and the Theorem 5 robustness value.

    export ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... OPENAI_API_KEY=...
    python gci_run.py --blocks gci/blocks.jsonl \
        --models anthropic:<claude-small> anthropic:<claude-large> google:<gemini-flash> google:<gemini-pro> \
                 openai:<gpt-small> openai:<gpt-large> \
        --origin Claude_Code=anthropic:<claude-large> Google_Jules=google:<gemini-pro> \
                 OpenAI_Codex=openai:<gpt-large> Copilot=openai:<gpt-large> \
        --theta 0.3 0.4 0.5 --R 3 --q 0.34 --out gci/

Per block:  control  = d^{(q)}(b; origin model) <= theta   (only when an origin proxy exists)
            swap     = min over models != origin of d^{(q)}(b; model)
            verdict  = under_specified | coupled | regenerable | no_control
Outputs:    verdicts.csv, per_repo_gci.csv, summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_oracle import LLMOracle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="model specs provider:model-id (the set M')")
    ap.add_argument("--origin", nargs="*", default=[], help="Agent=provider:model-id proxies for the control step")
    ap.add_argument("--theta", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    ap.add_argument("--delta", type=float, default=0.05, help="perturbation for the F(delta) report")
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.34)
    ap.add_argument("--distance", choices=["token", "alpha"], default="alpha",
                    help="token: strict token distance; alpha: same after renaming local names and the function name")
    ap.add_argument("--cache", default=None, help="cache file to reuse (default: <out>/cache.jsonl). Re-scoring a finished run with another --distance costs no API calls when pointed at its cache.")
    ap.add_argument("--prompt-level", choices=["pr", "doc"], default="doc",
                    help="pr: PR text only; doc: PR text + the function's own docstring as specification")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls (rate limits)")
    ap.add_argument("--per-repo", default=None, help="per_repo_metrics.csv to scale block-level rates to GCI")
    ap.add_argument("--out", default="gci")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blocks = [json.loads(l) for l in open(a.blocks)]
    origin = dict(kv.split("=", 1) for kv in a.origin)
    oracle = LLMOracle(a.cache or str(out / "cache.jsonl"), R=a.R, q=a.q, temperature=a.temperature, sleep=a.sleep, prompt_level=a.prompt_level, alpha=(a.distance == "alpha"))
    print(f"{len(blocks)} blocks, M' = {a.models}, origin proxies = {origin}")

    rows: List[Dict] = []
    for i, b in enumerate(blocks):
        d: Dict[str, float] = {m: oracle(b, m) for m in a.models}
        om = origin.get(b["agent"])
        d_self = oracle(b, om) if om else math.nan
        alts = {m: v for m, v in d.items() if m != om}
        row = {"block_id": b["block_id"], "repo_url": b["repo_url"], "agent": b["agent"], "n_lines": b["n_lines"],
               "origin_model": om or "", "d_self": d_self, "d_min_alt": min(alts.values()) if alts else math.nan,
               **{f"d[{m}]": v for m, v in d.items()}}
        rows.append(row)
        if i % 10 == 0:
            print(f"  block {i}/{len(blocks)}  api calls so far: {oracle.calls}", flush=True)
    V = pd.DataFrame(rows)

    summary: Dict = {"n_blocks": len(V), "models": a.models, "origin": origin, "R": a.R, "q": a.q,
                     "temperature": a.temperature, "prompt_level": a.prompt_level, "distance": a.distance,
                     "blocks_with_docstring": int(sum(1 for b in blocks if b.get("docstring"))),
                     "api_calls_this_run": oracle.calls,
                     "empty_answers_this_run": oracle.empty_answers,
                     "models_at_default_temperature": sorted(
                         f"{k}:{m}" for k, p in oracle._providers.items() for m in getattr(p, "no_temperature", [])),
                     "by_theta": {}}
    ctrl0 = V[V.origin_model != ""]
    if len(ctrl0):
        summary["d_self_controlled"] = {"n": int(len(ctrl0)),
            "quantiles_10_25_50_75_90": [round(float(ctrl0.d_self.quantile(p)), 3) for p in (.1, .25, .5, .75, .9)]}
    for theta in a.theta:
        def verdict(r):
            if r.origin_model == "":
                return "coupled_no_control" if r.d_min_alt > theta else "regenerable_no_control"
            if r.d_self > theta:
                return "under_specified"
            return "coupled" if r.d_min_alt > theta else "regenerable"
        V[f"verdict@{theta}"] = V.apply(verdict, axis=1)
        ctrl = V[V.origin_model != ""]
        res = {"verdicts": V[f"verdict@{theta}"].value_counts().to_dict(),
               "n_with_control": int(len(ctrl))}
        if len(ctrl):
            passed = ctrl[ctrl[f"verdict@{theta}"] != "under_specified"]
            res["under_specified_share"] = round(float((ctrl[f"verdict@{theta}"] == "under_specified").mean()), 3)
            res["coupled_block_share_given_control"] = round(float((passed[f"verdict@{theta}"] == "coupled").mean()), 3) if len(passed) else None
            res["coupled_line_share_given_control"] = round(float(passed.n_lines[passed[f"verdict@{theta}"] == "coupled"].sum() / passed.n_lines.sum()), 3) if len(passed) else None
        # Theorem 3(ii): nested alternative sets -> coupled share non-increasing
        mono = []
        for k in range(1, len(a.models) + 1):
            sub = a.models[:k]
            def coupled_k(r):
                alts = [r[f"d[{m}]"] for m in sub if m != r.origin_model]
                if not alts or (r.origin_model and r.d_self > theta):
                    return False
                return min(alts) > theta
            mono.append(round(float(V.apply(coupled_k, axis=1).mean()), 3))
        res["coupled_share_by_nested_M"] = mono
        res["theorem3_monotone"] = all(x >= y for x, y in zip(mono, mono[1:]))
        # Theorem 5: F(delta) on the min alternative distance
        res[f"F(delta={a.delta})"] = round(float(((V.d_min_alt >= theta - a.delta) & (V.d_min_alt <= theta + a.delta)).mean()), 3)
        summary["by_theta"][str(theta)] = res

    # per-repository GCI estimate at each theta
    per_repo_rows = []
    scale = None
    if a.per_repo:
        pr = pd.read_csv(a.per_repo).set_index("repo_url")
        scale = (pr.n_L + pr.n_P + pr.n_G) / pr.n_units   # share of the repository's units that are AI-generated
    for repo, g in V.groupby("repo_url"):
        r = {"repo_url": repo, "agent": g.agent.iloc[0], "n_blocks": len(g), "n_lines": int(g.n_lines.sum())}
        for theta in a.theta:
            coupled = g[f"verdict@{theta}"].isin(["coupled", "coupled_no_control"])
            share = float(g.n_lines[coupled].sum() / g.n_lines.sum())
            r[f"coupled_line_share@{theta}"] = round(share, 3)
            if scale is not None and repo in scale.index:
                r[f"GCI_est@{theta}"] = round(share * float(scale[repo]), 3)
        per_repo_rows.append(r)
    P = pd.DataFrame(per_repo_rows)
    V.to_csv(out / "verdicts.csv", index=False)
    P.to_csv(out / "per_repo_gci.csv", index=False)
    for theta in a.theta:
        col = f"GCI_est@{theta}" if f"GCI_est@{theta}" in P else f"coupled_line_share@{theta}"
        summary["by_theta"][str(theta)]["per_repo_median"] = round(float(P[col].median()), 3)
        summary["by_theta"][str(theta)]["per_repo_iqr"] = [round(float(P[col].quantile(.25)), 3), round(float(P[col].quantile(.75)), 3)]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
