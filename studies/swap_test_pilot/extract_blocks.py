"""Extract regeneration blocks (whole Python functions added by agent PRs) for the GCI study.

Reads the same AIDev tables as study_a_pipeline.py and the pooled repository list
produced by results/analyse.py, finds functions that a merged agent PR added in
full, and writes a stratified sample to blocks.jsonl.

    python studies/swap_test_pilot/extract_blocks.py --repos results/field_study/pooled_per_repo.csv --n-blocks 250 --out gci/blocks.jsonl

A block qualifies when every line of the function is an added ("+") line inside
one hunk, it has 5-80 logical lines, and it parses as Python on its own.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

HF_REPO = "hao-li/AIDev"
DEF_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(")


def table_path(name: str, revision: str, data_dir: Optional[str]) -> Path:
    if data_dir:
        return Path(data_dir) / f"{name}.parquet"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, f"{name}.parquet", repo_type="dataset", revision=revision))


def whole_functions(patch: str) -> List[Dict]:
    """Functions whose every line was added in one hunk."""
    out = []
    hunks = re.split(r"^@@.*$", patch, flags=re.M)
    for hunk in hunks:
        lines = hunk.splitlines()
        i = 0
        while i < len(lines):
            m = DEF_RE.match(lines[i][1:]) if lines[i].startswith("+") else None
            if not m:
                i += 1
                continue
            indent = len(m.group(1))
            body = [lines[i][1:]]
            j = i + 1
            while j < len(lines):
                l = lines[j]
                if not l.startswith("+"):
                    break
                src = l[1:]
                if src.strip() == "":
                    body.append(src); j += 1; continue
                cur = len(src) - len(src.lstrip())
                if cur <= indent:
                    break
                body.append(src); j += 1
            while body and body[-1].strip() == "":
                body.pop()
            code = "\n".join(l[indent:] if len(l) >= indent else l for l in body)
            logical = [l for l in body if l.strip() and not l.strip().startswith("#")]
            if 5 <= len(logical) <= 80:
                try:
                    tree = ast.parse(code)
                    fn = tree.body[0]
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sig = code.splitlines()[0].strip()
                        out.append({"func_name": m.group(2), "signature": sig, "code": code, "n_lines": len(logical),
                                    "docstring": ast.get_docstring(fn) or ""})
                except SyntaxError:
                    pass
            i = j
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", required=True, help="CSV with a repo_url column (e.g. results/field_study/pooled_per_repo.csv)")
    ap.add_argument("--scope", default="pop", choices=["pop", "all"])
    ap.add_argument("--revision", default="v5")
    ap.add_argument("--data-dir", default=os.environ.get("AIDEV_DATA_DIR"))
    ap.add_argument("--n-blocks", type=int, default=250)
    ap.add_argument("--max-per-repo", type=int, default=6)
    ap.add_argument("--max-prompt-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only-agents", nargs="*", default=[], help="keep only these agents (e.g. Claude_Code for the pilot)")
    ap.add_argument("--agent-quota", nargs="*", default=[],
                    help="Agent=N: take up to N blocks of this agent first, then fill the rest round-robin. "
                         "Use it for agents whose own vendor you can call (only they can be controlled).")
    ap.add_argument("--require-docstring", action="store_true", help="keep only functions that have a docstring")
    ap.add_argument("--out", default="gci/blocks.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    repos = set(pd.read_csv(a.repos)["repo_url"])

    pr_table = "pull_request" if a.scope == "pop" else "all_pull_request"
    prs = pq.read_table(table_path(pr_table, a.revision, a.data_dir),
                        columns=["id", "agent", "repo_url", "merged_at", "title", "body"]).to_pandas()
    prs = prs[prs["merged_at"].notna() & prs["repo_url"].isin(repos)]
    meta = prs.set_index("id")[["agent", "repo_url", "title", "body"]]
    ids = set(prs["id"])
    print(f"{len(ids):,} merged agent PRs in {len(repos)} repositories")

    candidates: List[Dict] = []
    pf = pq.ParquetFile(table_path("pr_commit_details", a.revision, a.data_dir))
    for i, batch in enumerate(pf.iter_batches(batch_size=32_768, columns=["pr_id", "sha", "filename", "patch"])):
        df = batch.to_pandas()
        hit = df[df["pr_id"].isin(ids) & df["filename"].astype(str).str.endswith(".py") & df["patch"].notna()]
        for row in hit.itertuples(index=False):
            for fn in whole_functions(row.patch):
                m = meta.loc[row.pr_id]
                prompt = f"{m.title or ''}\n\n{m.body or ''}"[: a.max_prompt_chars]
                candidates.append({"pr_id": int(row.pr_id), "sha": row.sha, "filename": row.filename,
                                   "agent": m.agent, "repo_url": m.repo_url, "prompt": prompt, **fn})
        if i % 100 == 0:
            print(f"  batch {i}: {len(candidates):,} candidate blocks", flush=True)
    if a.only_agents:
        candidates = [c for c in candidates if c["agent"] in a.only_agents]
    if a.require_docstring:
        candidates = [c for c in candidates if c["docstring"]]
    print(f"{len(candidates):,} candidate whole-function blocks "
          f"({sum(1 for c in candidates if c['docstring']):,} with a docstring)")

    # stratify: cap per repository, then round-robin across agents
    by_repo: Dict[str, List[Dict]] = {}
    for c in candidates:
        by_repo.setdefault(c["repo_url"], []).append(c)
    pools: Dict[str, List[Dict]] = {}
    for repo, items in by_repo.items():
        rng.shuffle(items)
        for c in items[: a.max_per_repo]:
            pools.setdefault(c["agent"], []).append(c)
    for p in pools.values():
        rng.shuffle(p)
    agents = sorted(pools, key=lambda ag: -len(pools[ag]))
    sample: List[Dict] = []
    for kv in a.agent_quota:
        ag, n = kv.split("=", 1)
        take = min(int(n), len(pools.get(ag, [])), a.n_blocks)
        sample += [pools[ag].pop() for _ in range(take)]
        print(f"  quota {ag}: {take} of {n} requested ({take} available up to the quota)")
    while len(sample) < a.n_blocks and any(pools.values()):
        for ag in agents:
            if pools[ag] and len(sample) < a.n_blocks:
                sample.append(pools[ag].pop())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for k, c in enumerate(sample):
            c["block_id"] = f"b{k:04d}"
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(sample)} blocks to {out}; by agent: "
          f"{pd.Series([c['agent'] for c in sample]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
