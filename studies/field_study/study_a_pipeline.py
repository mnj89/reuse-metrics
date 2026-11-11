"""Study A: provenance metrics on agent-authored pull requests (AIDev).

Runs on Colab or a workstation against the AIDev-pop subset (repositories with
> 100 stars). Follows the dataset authors' memory-safe loading pattern.

Usage (Colab):
    !pip install reuse-metrics pandas pyarrow huggingface_hub
    !python study_a_pipeline.py --scope pop --n-repos 40 --seed 1 --out output/

Usage (local parquet copies):
    AIDEV_DATA_DIR=/data/aidev python study_a_pipeline.py --n-repos 40

What it computes per sampled repository
    units   = logical lines ADDED by merged agent PRs in Python files
    event   = one PR (prompt proxy = title + body, model = agent name)
    T1 (D)  = import / from-import lines and lines carrying an attribution comment
    T3b (P) = PR text names a pattern in studies/field_study/patterns.txt
    T3c (L) = otherwise
    T3a (G) = only if --oracle is given (see --help); otherwise reported as "not run"
    H       = lines added by human PRs in the same repos, if --human-prs is supplied
              (a parquet with the same schema; AIDev holds agent PRs only)

Outputs
    output/units.parquet          every unit with its class and PR id
    output/per_repo_metrics.csv   DRR, LRD(log-based), PPC, PPC_U, PLC, GCI, HRS per repo
    output/summary.json           medians, IQRs, parameter tuple, pattern-file hash
    output/table_rq1.tex          LaTeX rows for the empirical-section template
    output/equal_drr_pairs.csv    repository pairs with |DRR diff| < 0.01 but different profiles (Theorem 2)

Nothing in this script invents data: every number comes from the dataset rows read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from reuse_metrics import Attribution, Event, Prov, System, Unit, counts, profile  # noqa: E402
from reuse_metrics.metrics import component_vector, drr, gci_true, hrs, information_loss, lrd_true, ppc, ppc_units  # noqa: E402

HF_REPO = "hao-li/AIDev"
IMPORT_RE = re.compile(r"^\s*(import\s+\w|from\s+[\w.]+\s+import\s)")
ATTRIB_RE = re.compile(r"(adapted from|copied from|source:|taken from|based on)\s*https?://", re.I)
COMMENT_OR_BLANK = re.compile(r"^\s*(#.*)?$")


# ----------------------------------------------------------------------------- data access

def table_path(name: str, revision: str, data_dir: Optional[str]) -> Path:
    if data_dir:
        return Path(data_dir) / f"{name}.parquet"
    from huggingface_hub import hf_hub_download  # imported lazily: not needed with local copies
    return Path(hf_hub_download(HF_REPO, f"{name}.parquet", repo_type="dataset", revision=revision))


def load_patterns(path: Path) -> Set[str]:
    return {l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}


def added_lines(patch: str) -> List[str]:
    """Logical lines added by a unified diff hunk (no blank lines, no pure comments)."""
    out = []
    for raw in patch.splitlines():
        if raw.startswith("+") and not raw.startswith("+++"):
            line = raw[1:]
            if not COMMENT_OR_BLANK.match(line):
                out.append(line)
    return out


# ----------------------------------------------------------------------------- core

def build_system(pr_rows: pd.DataFrame, commit_rows: pd.DataFrame, human_rows: Optional[pd.DataFrame],
                 human_commits: Optional[pd.DataFrame], patterns: Set[str]) -> tuple:
    """Build a `System` for one repository from AIDev rows. Returns (system, partition_no_G)."""
    units: List[Unit] = []
    attributions: Dict[tuple, Attribution] = {}
    events: List[Event] = []
    part: Dict[tuple, Prov] = {}

    def add_pr_units(prs: pd.DataFrame, commits: pd.DataFrame, is_agent: bool) -> None:
        by_pr = commits.groupby("pr_id")
        for pr in prs.itertuples(index=False):
            if pr.id not in by_pr.groups:
                continue
            keys = set(); emitted = {}; block_of = {}
            prompt = f"{pr.title or ''}\n{pr.body or ''}"
            for c in by_pr.get_group(pr.id).itertuples(index=False):
                if not str(c.filename).endswith(".py") or not isinstance(c.patch, str):
                    continue
                fn_name = None
                for i, line in enumerate(added_lines(c.patch)):
                    m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", line)
                    if m:
                        fn_name = m.group(1)
                    u = Unit(c.filename, i, c.sha, line)
                    if u.key in part:
                        continue
                    units.append(u)
                    if IMPORT_RE.match(line) or ATTRIB_RE.search(line):
                        attributions[u.key] = Attribution("import" if IMPORT_RE.match(line) else "copy-attributed",
                                                          "import" if IMPORT_RE.match(line) else "copy")
                        part[u.key] = Prov.D
                    elif is_agent:
                        keys.add(u.key); emitted[u.key] = line; block_of[u.key] = fn_name
                        named = any(p.lower() in prompt.lower() for p in patterns)
                        part[u.key] = Prov.P if named else Prov.L
                    else:
                        part[u.key] = Prov.H
            if is_agent and keys:
                events.append(Event(str(pr.id), 0.0, prompt, str(pr.agent), frozenset(keys),
                                    prompt_stored=bool(pr.title or pr.body), emitted_text=emitted, block_of=block_of))

    add_pr_units(pr_rows, commit_rows, True)
    if human_rows is not None and human_commits is not None:
        add_pr_units(human_rows, human_commits, False)
    return System(units, attributions, events), part


def repo_metrics(system: System, part: Dict[tuple, Prov], patterns: Set[str], beta) -> Dict[str, float]:
    n = len(part)
    if n == 0:
        return {}
    c = counts(part)
    plc = (sum(1 for e in system.events if e.prompt_stored) / len(system.events)) if system.events else 0.0
    d, l, g = drr(part), lrd_true(part), gci_true(part)
    p = ppc(system.events, patterns)
    # PPA needs prompt replay (PRF); not available offline -> report PLC only, PPA term set to PLC
    v = component_vector(d, l, p, plc, g)
    loss, ceil = information_loss(profile(part))
    return {"n_units": n, "n_events": len(system.events), **{f"n_{k.value}": v_ for k, v_ in c.items()},
            "DRR": d, "LRD_log": l, "PPC": p, "PPC_U": ppc_units(system.events, patterns),
            "PLC": plc, "GCI": g, "HRS_equal_weights": hrs(v, beta), "info_loss_bits": loss}


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", default="pop", choices=["pop", "all"])
    ap.add_argument("--revision", default="v5")
    ap.add_argument("--data-dir", default=os.environ.get("AIDEV_DATA_DIR"))
    ap.add_argument("--n-repos", type=int, default=40)
    ap.add_argument("--max-prs-per-repo", type=int, default=30)
    ap.add_argument("--min-prs", type=int, default=5, help="min merged Python-touching agent PRs for a repo to be eligible")
    ap.add_argument("--agents", nargs="*", default=None, help="restrict to these agent names")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--patterns", default=str(Path(__file__).with_name("patterns.txt")))
    ap.add_argument("--human-prs", default=None, help="parquet of human PRs (same schema) for the H comparison set")
    ap.add_argument("--human-commits", default=None, help="parquet of their commit details")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns(Path(a.patterns))
    pat_hash = hashlib.sha256(Path(a.patterns).read_bytes()).hexdigest()[:12]
    pr_table = "pull_request" if a.scope == "pop" else "all_pull_request"

    # 1. merged agent PRs, light columns
    prs = pq.read_table(table_path(pr_table, a.revision, a.data_dir),
                        columns=["id", "number", "agent", "repo_url", "merged_at", "title", "body"]).to_pandas()
    prs = prs[prs["merged_at"].notna()]
    if a.agents:
        prs = prs[prs["agent"].isin(a.agents)]
    print(f"{len(prs):,} merged agent PRs; agents: {prs['agent'].value_counts().to_dict()}")

    # 2. first pass over commit details (two light columns): which merged PRs touch Python?
    commit_path = table_path("pr_commit_details", a.revision, a.data_dir)
    merged_ids = set(prs["id"].tolist())
    py_prs: Set = set()
    pf = pq.ParquetFile(commit_path)
    n_rg = pf.metadata.num_row_groups
    for i, batch in enumerate(pf.iter_batches(batch_size=262_144, columns=["pr_id", "filename"])):
        df = batch.to_pandas()
        hit = df[df["pr_id"].isin(merged_ids) & df["filename"].astype(str).str.endswith(".py")]
        py_prs.update(hit["pr_id"].unique().tolist())
        if i % 20 == 0:
            print(f"  pass 1: batch {i}, {len(py_prs):,} Python-touching PRs so far", flush=True)
    prs = prs[prs["id"].isin(py_prs)]
    print(f"{len(prs):,} merged agent PRs touch at least one Python file")

    # 3. stratified repository sample among repos with >= --min-prs eligible PRs
    eligible = prs.groupby("repo_url").size()
    eligible = eligible[eligible >= a.min_prs]
    prs = prs[prs["repo_url"].isin(eligible.index)]
    repo_agent = prs.groupby("repo_url")["agent"].agg(lambda s: s.value_counts().index[0])
    print(f"{len(eligible):,} repositories with >= {a.min_prs} such PRs; by dominant agent: "
          f"{repo_agent.value_counts().to_dict()}")
    agents = sorted(repo_agent.unique(), key=lambda ag: -(repo_agent == ag).sum())
    pools = {ag: list(repo_agent[repo_agent == ag].index) for ag in agents}
    for p in pools.values():
        rng.shuffle(p)
    sample: List[str] = []
    while len(sample) < a.n_repos and any(pools.values()):   # round-robin across agents
        for ag in agents:
            if pools[ag] and len(sample) < a.n_repos:
                sample.append(pools[ag].pop())
    prs = prs[prs["repo_url"].isin(sample)]
    prs = pd.concat([g.sample(min(len(g), a.max_prs_per_repo), random_state=a.seed)
                     for _, g in prs.groupby("repo_url")], ignore_index=True)
    print(f"sampled {len(sample)} repositories, {len(prs):,} PRs")

    # 4. second pass: read the patches of the sampled PR ids only
    ids = set(prs["id"].tolist())
    chunks = []
    for i, batch in enumerate(pf.iter_batches(batch_size=32_768, columns=["pr_id", "sha", "filename", "status", "patch"])):
        df = batch.to_pandas()
        hit = df[df["pr_id"].isin(ids) & df["filename"].astype(str).str.endswith(".py")]
        if len(hit):
            chunks.append(hit)
        if i % 100 == 0:
            print(f"  pass 2: batch {i}, {sum(len(c) for c in chunks):,} file changes collected", flush=True)
    commits = pd.concat(chunks) if chunks else pd.DataFrame(columns=["pr_id", "sha", "filename", "status", "patch"])
    print(f"{len(commits):,} Python file changes read")

    human_prs = pd.read_parquet(a.human_prs) if a.human_prs else None
    human_commits = pd.read_parquet(a.human_commits) if a.human_commits else None

    # 5. per-repository systems and metrics
    beta = (0.2, 0.2, 0.2, 0.2, 0.2)
    rows, unit_rows = [], []
    for repo, g in prs.groupby("repo_url"):
        hp = human_prs[human_prs["repo_url"] == repo] if human_prs is not None else None
        system, part = build_system(g, commits, hp, human_commits, patterns)
        m = repo_metrics(system, part, patterns, beta)
        if not m:
            continue
        rows.append({"repo_url": repo, "agent": repo_agent[repo], **m})
        unit_rows += [{"repo_url": repo, "file": k[0], "line": k[1], "sha": k[2], "class": c.value} for k, c in part.items()]
    per_repo = pd.DataFrame(rows)
    per_repo.to_csv(out / "per_repo_metrics.csv", index=False)
    pd.DataFrame(unit_rows).to_parquet(out / "units.parquet", index=False)

    # 6. summary, LaTeX rows, Theorem-2 pairs
    metric_cols = ["DRR", "LRD_log", "PPC", "PPC_U", "PLC", "GCI", "HRS_equal_weights"]
    summary = {
        "n_repos": int(len(per_repo)), "n_units": int(per_repo["n_units"].sum()),
        "n_events": int(per_repo["n_events"].sum()),
        "median": per_repo[metric_cols].median().round(3).to_dict(),
        "iqr": (per_repo[metric_cols].quantile(.75) - per_repo[metric_cols].quantile(.25)).round(3).to_dict(),
        "params": {"rho0": "n/a (post-merge edits not observed)", "N_file_sha256": pat_hash,
                   "granularity": "logical line added in merged PR, Python files",
                   "GCI": "not run (requires a regeneration oracle)", "H_set": bool(a.human_prs)},
        "scope": a.scope, "revision": a.revision, "seed": a.seed, "min_prs": a.min_prs, "max_prs_per_repo": a.max_prs_per_repo,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "table_rq1.tex").open("w") as f:
        med, iqr = summary["median"], summary["iqr"]
        cell = lambda k: f"{med[k]:.2f} [{iqr[k]:.2f}]"
        f.write("Study A, agent PRs & " + " & ".join(cell(k) for k in ["DRR", "LRD_log", "PPC", "PLC", "GCI", "HRS_equal_weights"]) + " \\\\\n")
    pairs = []
    pr_ = per_repo.reset_index(drop=True)
    for i in range(len(pr_)):
        for j in range(i + 1, len(pr_)):
            if abs(pr_.DRR[i] - pr_.DRR[j]) < 0.01 and (abs(pr_.LRD_log[i] - pr_.LRD_log[j]) > 0.1 or abs(pr_.PPC[i] - pr_.PPC[j]) > 0.2):
                pairs.append({"repo_a": pr_.repo_url[i], "repo_b": pr_.repo_url[j], "DRR": round(pr_.DRR[i], 3),
                              "LRD_a": round(pr_.LRD_log[i], 3), "LRD_b": round(pr_.LRD_log[j], 3),
                              "PPC_a": round(pr_.PPC[i], 3), "PPC_b": round(pr_.PPC[j], 3)})
    pd.DataFrame(pairs).to_csv(out / "equal_drr_pairs.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
