"""Pools the field-study runs and writes every number of Section IX (results.json, pooled_per_repo.csv).

    python studies/field_study/analyse.py            # reads and writes results/field_study/

The figure of the section is drawn by figures/figs_data.py from pooled_per_repo.csv.
"""
import json, sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results" / "field_study"
RUNS = {"seed1_min5": "seed 1, >= 5 PRs", "seed2_min5": "seed 2, >= 5 PRs", "seed3_min5": "seed 3, >= 5 PRs", "seed1_min10": "seed 1, >= 10 PRs"}
runs = {k: pd.read_csv(ROOT / "runs" / k / "per_repo_metrics.csv") for k in RUNS}
for k, d in runs.items(): d["run"] = k
# pooled: seeds 1-3 (identical parameters), first occurrence of each repo
pooled = pd.concat([runs["seed1_min5"], runs["seed2_min5"], runs["seed3_min5"]]).drop_duplicates("repo_url").reset_index(drop=True)
pooled["repo"] = pooled.repo_url.str.replace("https://api.github.com/repos/", "")
M = ["DRR","LRD_log","PPC","PPC_U","PLC","HRS_equal_weights"]
def q(d): return pd.DataFrame({"median": d[M].median(), "q1": d[M].quantile(.25), "q3": d[M].quantile(.75), "min": d[M].min(), "max": d[M].max()}).round(3)
out = {}
out["pooled_n_repos"] = int(len(pooled)); out["pooled_n_units"] = int(pooled.n_units.sum()); out["pooled_n_events"] = int(pooled.n_events.sum())
out["pooled_units_by_class"] = {c: int(pooled[f"n_{c}"].sum()) for c in "DLP"}
out["pooled_stats"] = q(pooled).to_dict()
out["agents"] = pooled.agent.value_counts().to_dict()
out["per_agent_median"] = pooled.groupby("agent")[["DRR","LRD_log","PPC","HRS_equal_weights"]].median().round(3).to_dict(orient="index")
out["per_agent_n"] = pooled.groupby("agent").size().to_dict()
out["per_run_median"] = {k: d[M].median().round(3).to_dict() for k, d in runs.items()}
out["per_run_n"] = {k: int(len(d)) for k, d in runs.items()}
# Theorem 2 empirical counterpart on pooled sample
pairs = []
P = pooled.reset_index(drop=True)
for i in range(len(P)):
    for j in range(i+1, len(P)):
        if abs(P.DRR[i]-P.DRR[j]) < 0.01:
            pairs.append((abs(P.LRD_log[i]-P.LRD_log[j]), abs(P.PPC[i]-P.PPC[j])))
pairs = np.array(pairs)
out["equal_drr_pairs_total"] = int(len(pairs))
out["equal_drr_pairs_lrd_gt_0.1"] = int((pairs[:,0] > 0.1).sum())
out["equal_drr_pairs_lrd_gt_0.3"] = int((pairs[:,0] > 0.3).sum())
out["equal_drr_pairs_ppc_gt_0.2"] = int((pairs[:,1] > 0.2).sum())
out["max_lrd_gap_at_equal_drr"] = float(pairs[:,0].max())
# correlations
from scipy.stats import spearmanr, kruskal
out["spearman_LRD_PPC"] = [round(x,3) for x in spearmanr(pooled.LRD_log, pooled.PPC)]
out["spearman_DRR_units"] = [round(x,3) for x in spearmanr(pooled.DRR, pooled.n_units)]
out["spearman_LRD_units"] = [round(x,3) for x in spearmanr(pooled.LRD_log, pooled.n_units)]
groups = [g.LRD_log.values for _, g in pooled.groupby("agent")]
out["kruskal_LRD_by_agent"] = [round(x,3) for x in kruskal(*groups)]
groups = [g.PPC.values for _, g in pooled.groupby("agent")]
out["kruskal_PPC_by_agent"] = [round(x,3) for x in kruskal(*groups)]
# pooled minimum DRR example rows for text
out["drr_zero_repos"] = int((pooled.DRR == 0).sum()); out["ppc_zero_repos"] = int((pooled.PPC == 0).sum())
out["ppc_zero_share"] = round(float((pooled.PPC == 0).mean()), 3)
json.dump(out, open(ROOT / "results.json", "w"), indent=2)
pooled.to_csv(ROOT / "pooled_per_repo.csv", index=False)
print(json.dumps(out, indent=2))
