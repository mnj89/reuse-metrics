"""Turn a GCI run (gci_run.py output) into the paper's results: tables, figures
and one JSON file with every number quoted in the text.

    python gci_analyse.py --run gci --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5 glm:glm-4.7-flash \
        --vendors anthropic=Anthropic glm=Z.ai --theta-main 0.4 --out gci/analysis

Reads   <run>/verdicts.csv, <run>/summary.json, <run>/per_repo_gci.csv
Writes  <out>/results.json          every number, with 95% cluster-bootstrap intervals (repositories as clusters)
        <out>/tab_gci_overview.tex  verdicts and coupled shares by group and theta
        <out>/tab_gci_agents.tex    coupled share per agent at the main theta
        <out>/tab_gci_models.tex    per-model regeneration rates
        <out>/section_gci_draft.tex draft Section IX subsection with the numbers filled in (edit before use)
        <out>/fig_gci_distances.pdf distance distributions per alternative model, theta marked
        <out>/fig_gci_nested.pdf    coupled share as alternative models are added (Theorem 3), per theta
        <out>/fig_gci_agents.pdf    coupled share per agent with intervals
        <out>/fig_gci_robustness.pdf F(delta) for the three theta values (Theorem 5)
        (.png copies of every figure for quick viewing)

Interpretation notes written into results.json:
  * The nested-set curve is non-increasing by construction (a minimum over a larger set
    cannot be larger), which is what Theorem 3(ii) states; the empirical content is the
    SIZE of the drop, in particular when the first cross-vendor model is added.
  * "Controlled" = blocks whose agent has an origin proxy (here Claude Code). Only these
    pass through the self-regeneration control, so only these estimate GCI as defined.
    Other agents' blocks are a swap-only comparison and are reported separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#1f5fa8", "#c8641c", "#2f8f5b", "#7a3fa0", "#9a9a9a"]


# ----------------------------------------------------------------------------- helpers

def cluster_boot(df: pd.DataFrame, stat, cluster: str = "repo_url", n: int = 2000, seed: int = 0):
    """Point estimate and 95% percentile interval, resampling whole repositories."""
    if df.empty:
        return [math.nan, math.nan, math.nan]
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster)]
    point = stat(df)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    vals = np.array([v for v in vals if not math.isnan(v)])
    if not len(vals):
        return [round(point, 3), math.nan, math.nan]
    return [round(point, 3), round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def line_share(df: pd.DataFrame, coupled_mask) -> float:
    tot = df.n_lines.sum()
    return float(df.n_lines[coupled_mask(df)].sum() / tot) if tot else math.nan


def fmt_ci(v) -> str:
    p, lo, hi = v
    if any(isinstance(x, float) and math.isnan(x) for x in v):
        return "--"
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


def short_model(m: str) -> str:
    return m.split(":", 1)[1] if ":" in m else m


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of gci_run.py")
    ap.add_argument("--models", nargs="+", required=True, help="the same --models list, in the same order")
    ap.add_argument("--vendors", nargs="*", default=[], help="provider-or-model-prefix=Vendor name, e.g. anthropic=Anthropic ollama:qwen3.8=Alibaba ollama:nemotron=NVIDIA")
    ap.add_argument("--theta-main", type=float, default=0.4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    out = Path(a.out) if a.out else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    V = pd.read_csv(run / "verdicts.csv")
    S = json.loads((run / "summary.json").read_text())
    thetas = sorted(float(t) for t in S["by_theta"])
    th0 = min(thetas, key=lambda t: abs(t - a.theta_main))
    vendor_of = dict(kv.split("=", 1) for kv in a.vendors)
    def vend(m):
        # longest key that prefixes the model spec wins: "ollama:qwen=Alibaba" beats "ollama=Ollama"
        for k in sorted(vendor_of, key=len, reverse=True):
            if m == k or m.startswith(k + ":") or m.startswith(k):
                return vendor_of[k]
        return m.split(":", 1)[0]
    V["origin_model"] = V["origin_model"].fillna("")
    V["controlled"] = V.origin_model != ""
    R: Dict = {"n_blocks": int(len(V)), "n_repos": int(V.repo_url.nunique()),
               "n_lines": int(V.n_lines.sum()), "models": a.models,
               "vendors": {m: vend(m) for m in a.models}, "theta_values": thetas, "theta_main": th0,
               "R": S.get("R"), "q": S.get("q"), "temperature": S.get("temperature"),
               "models_at_default_temperature": S.get("models_at_default_temperature", []),
               "empty_answers": S.get("empty_answers_this_run", {}),
               "blocks_by_agent": V.agent.value_counts().to_dict(),
               "controlled_blocks": int(V.controlled.sum()), "by_theta": {}}

    # ---------------------------------------------------------------- per theta
    for th in thetas:
        col = f"verdict@{th}"
        C = V[V.controlled]
        U = V[~V.controlled]
        passed = C[C[col] != "under_specified"]
        res = {
            "verdict_counts": V[col].value_counts().to_dict(),
            "under_specified_share_controlled": cluster_boot(C, lambda d: float((d[col] == "under_specified").mean()), n=a.boot),
            "coupled_block_share_controlled": cluster_boot(passed, lambda d: float((d[col] == "coupled").mean()), n=a.boot),
            "coupled_line_share_controlled": cluster_boot(passed, lambda d: line_share(d, lambda x: x[col] == "coupled"), n=a.boot),
            "coupled_block_share_uncontrolled": cluster_boot(U, lambda d: float((d[col] == "coupled_no_control").mean()), n=a.boot),
            "coupled_line_share_uncontrolled": cluster_boot(U, lambda d: line_share(d, lambda x: x[col] == "coupled_no_control"), n=a.boot),
        }
        # nested alternative sets, in --models order (Theorem 3(ii)); separately per group
        nested = {}
        for grp, D in [("controlled", passed), ("uncontrolled", U)]:
            curve = []
            for k in range(1, len(a.models) + 1):
                sub = a.models[:k]
                def coupled_k(r, sub=sub):
                    alts = [r[f"d[{m}]"] for m in sub if m != r.origin_model]
                    return bool(alts) and min(alts) > th
                curve.append(round(float(D.apply(coupled_k, axis=1).mean()), 3) if len(D) else math.nan)
            nested[grp] = curve
        res["coupled_share_nested"] = nested
        # within-vendor vs cross-vendor alternatives
        cv = {}
        for grp, D in [("controlled", passed), ("uncontrolled", U)]:
            if not len(D):
                continue
            def share(mask_models):
                def f(r):
                    alts = [r[f"d[{m}]"] for m in a.models if m != r.origin_model and mask_models(m, r)]
                    return bool(alts) and min(alts) > th
                return round(float(D.apply(f, axis=1).mean()), 3)
            first_vendor = vend(a.models[0])
            cv[grp] = {"same_vendor_as_first_model_only": share(lambda m, r: vend(m) == first_vendor),
                       "all_models": share(lambda m, r: True)}
        res["within_vs_cross_vendor"] = cv
        # per-model regeneration rate (share of blocks the model gets within theta), excluding its own origin blocks
        res["regeneration_rate"] = {m: round(float((V[V.origin_model != m][f"d[{m}]"] <= th).mean()), 3) for m in a.models}
        # robustness F(delta) curve (Theorem 5)
        res["F_delta"] = {str(d): round(float(((V.d_min_alt >= th - d) & (V.d_min_alt <= th + d)).mean()), 3)
                          for d in (0.01, 0.025, 0.05, 0.1)}
        R["by_theta"][str(th)] = res

    # per agent at main theta
    col0 = f"verdict@{th0}"
    agents = {}
    for ag, D in V.groupby("agent"):
        ctrl = bool(D.controlled.any())
        DD = D[D[col0] != "under_specified"] if ctrl else D
        lab = "coupled" if ctrl else "coupled_no_control"
        agents[ag] = {"n_blocks": int(len(D)), "controlled": ctrl,
                      "under_specified_share": round(float((D[col0] == "under_specified").mean()), 3) if ctrl else None,
                      "coupled_block_share": cluster_boot(DD, lambda d: float((d[col0] == lab).mean()), n=a.boot)}
    R["per_agent_at_theta_main"] = agents
    # per repository GCI distribution
    P = pd.read_csv(run / "per_repo_gci.csv")
    gcol = f"GCI_est@{th0}" if f"GCI_est@{th0}" in P else f"coupled_line_share@{th0}"
    R["per_repo"] = {"column": gcol, "n": int(len(P)), "median": round(float(P[gcol].median()), 3),
                     "q1": round(float(P[gcol].quantile(.25)), 3), "q3": round(float(P[gcol].quantile(.75)), 3),
                     "share_of_repos_with_any_coupled_block": round(float((P[gcol] > 0).mean()), 3)}
    R["notes"] = [
        "The nested-set curve is non-increasing by construction (Theorem 3(ii)); the empirical content is the size of each drop.",
        "Controlled blocks (agent with an origin proxy) estimate GCI as defined; uncontrolled blocks are a swap-only comparison.",
        "Intervals are 95% percentile intervals from a bootstrap over repositories (blocks within a repository resampled together).",
    ]
    (out / "results.json").write_text(json.dumps(R, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))

    # ---------------------------------------------------------------- tables
    with (out / "tab_gci_overview.tex").open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\n & " + " & ".join(f"$\\theta={t:g}$" for t in thetas) + " \\\\\n\\midrule\n")
        rows = [("Controlled: under-specified", "under_specified_share_controlled"),
                ("Controlled: coupled blocks", "coupled_block_share_controlled"),
                ("Controlled: coupled lines", "coupled_line_share_controlled"),
                ("Swap only: coupled blocks", "coupled_block_share_uncontrolled"),
                ("Swap only: coupled lines", "coupled_line_share_uncontrolled")]
        for lab, key in rows:
            f.write(lab + " & " + " & ".join(fmt_ci(R["by_theta"][str(t)][key]) for t in thetas) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with (out / "tab_gci_agents.tex").open("w") as f:
        f.write("\\begin{tabular}{lccc}\n\\toprule\nAgent & Blocks & Control & Coupled share (95\\% CI) \\\\\n\\midrule\n")
        for ag, r in sorted(agents.items(), key=lambda kv: -kv[1]["n_blocks"]):
            f.write(f"{ag.replace('_', ' ')} & {r['n_blocks']} & {'yes' if r['controlled'] else 'no'} & {fmt_ci(r['coupled_block_share'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with (out / "tab_gci_models.tex").open("w") as f:
        f.write("\\begin{tabular}{llc}\n\\toprule\nModel & Vendor & Regenerates (share of blocks, $\\theta=" + f"{th0:g}" + "$) \\\\\n\\midrule\n")
        for m in a.models:
            f.write(f"\\texttt{{{short_model(m)}}} & {vend(m)} & {R['by_theta'][str(th0)]['regeneration_rate'][m]:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------- figures
    # 1. distance distributions
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bins = np.linspace(0, 1, 26)
    for i, m in enumerate(a.models):
        d = V[V.origin_model != m][f"d[{m}]"].dropna()
        ax.hist(d, bins=bins, histtype="step", lw=1.6, color=PALETTE[i % 5], label=f"{short_model(m)} ({vend(m)})")
    for t in thetas:
        ax.axvline(t, color="#555", ls=":" if t != th0 else "--", lw=1)
    ax.set_xlabel("Distance of the closest of R regenerations to the original block")
    ax.set_ylabel("Blocks"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Regeneration distance by model (dotted/dashed: $\\theta$ values)", fontsize=9.5, fontweight="bold")
    save(fig, out, "fig_gci_distances")

    # 2. nested curve (Theorem 3)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    xlabels = [short_model(m) for m in a.models]
    origin_names = sorted({short_model(m) for m in V.origin_model.unique() if m})
    ctl_title = "Controlled blocks" + (f"\n(origin {', '.join(origin_names)} skipped)" if origin_names else "")
    for ax, grp, title in [(axs[0], "controlled", ctl_title), (axs[1], "uncontrolled", "Swap-only blocks\n(other agents)")]:
        for i, t in enumerate(thetas):
            y = R["by_theta"][str(t)]["coupled_share_nested"][grp]
            ax.plot(range(1, len(y) + 1), y, "-o", color=PALETTE[i], ms=4, label=f"$\\theta={t:g}$")
        ax.set_xticks(range(1, len(a.models) + 1))
        ax.set_xticklabels(["+" + x if k else x for k, x in enumerate(xlabels)], rotation=20, ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("Share of blocks coupled"); axs[1].legend(frameon=False, fontsize=8)
    save(fig, out, "fig_gci_nested")

    # 3. per agent
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    items = sorted(agents.items(), key=lambda kv: (not kv[1]["controlled"], kv[0]))
    for i, (ag, r) in enumerate(items):
        p, lo, hi = r["coupled_block_share"]
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        c = PALETTE[0] if r["controlled"] else PALETTE[4]
        ax.bar(i, p, color=c, width=0.6)
        if not (isinstance(lo, float) and math.isnan(lo)):
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="k", capsize=3, lw=1)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([ag.replace("_", " ") for ag, _ in items], fontsize=8)
    ax.set_ylabel(f"Coupled share ($\\theta={th0:g}$)"); ax.set_ylim(0, 1.02)
    ax.set_title("Coupled blocks by agent (blue: with control step; grey: swap only)", fontsize=9.5, fontweight="bold")
    save(fig, out, "fig_gci_agents")

    # 4. robustness
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    deltas = np.linspace(0, 0.15, 31)
    for i, t in enumerate(thetas):
        F = [float(((V.d_min_alt >= t - d) & (V.d_min_alt <= t + d)).mean()) for d in deltas]
        ax.plot(deltas, F, color=PALETTE[i], lw=1.6, label=f"$\\theta={t:g}$")
    ax.set_xlabel("$\\delta$"); ax.set_ylabel("$F(\\delta)$: max change in GCI")
    ax.set_title("Robustness bound (Theorem 5)", fontsize=9.5, fontweight="bold"); ax.legend(frameon=False, fontsize=8)
    save(fig, out, "fig_gci_robustness")

    # ---------------------------------------------------------------- draft text (numbers filled in; edit before use)
    m0 = R["by_theta"][str(th0)]
    ci = lambda v: f"{v[0]:.2f} (95\\% CI {v[1]:.2f}--{v[2]:.2f})" if v and not any(isinstance(x, float) and math.isnan(x) for x in v) else "n/a"
    names = ", ".join(f"\\texttt{{{short_model(m)}}} ({vend(m)})" for m in a.models)
    dt = R["models_at_default_temperature"]
    default_temp_sentence = ("These models refused a custom temperature and ran at their default: "
                             + ", ".join("\\texttt{" + short_model(m) + "}" for m in dt) + ".") if dt else ""
    cvc = m0["within_vs_cross_vendor"].get("controlled", {})
    cvu = m0["within_vs_cross_vendor"].get("uncontrolled", {})
    agent_counts = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in R["blocks_by_agent"].items())
    dist_name = S.get("distance", "token")
    dist_sentence = ("a normalised token distance after AST normalisation" if dist_name == "token" else
                     "a normalised token distance after AST normalisation and renaming of local names (\\emph{alpha-renamed})")
    if S.get("prompt_level") == "doc":
        dist_sentence += ", with the PR text and the function's own docstring as the prompt"
    tex = f"""% Draft generated by gci_analyse.py from {run}/ -- check every sentence against results.json before use.
\\subsection{{Generative Coupling}}\\label{{sec:emp-gci}}
\\textbf{{Setup.}} We sampled {R['n_blocks']} whole functions added by merged agent pull requests in {R['n_repos']} of the study repositories
({agent_counts}) and ran the model-swap test of Section~\\ref{{sec:opsswap}} with
the alternative-model set $M'$ consisting of {names}; $R = {R['R']}$ samples per model, $q = {R['q']}$, and $\\theta \\in \\{{{', '.join(f'{t:g}' for t in thetas)}\\}}$ on
{dist_sentence}. Only Claude Code has its vendor in $M'$, so its {R['controlled_blocks']} blocks pass through the
self-regeneration control with \\texttt{{{', '.join(origin_names)}}} as proxy for the originating model; the other agents' blocks are reported
as a swap-only comparison. {default_temp_sentence}

\\textbf{{Results.}} At $\\theta = {th0:g}$, {ci(m0['under_specified_share_controlled'])} of controlled blocks were under-specified, i.e.\\ the
pull-request text did not let the originating vendor's model reproduce its own function. Of the remaining controlled blocks,
{ci(m0['coupled_block_share_controlled'])} were coupled: no alternative model regenerated them. Among swap-only blocks the coupled share was
{ci(m0['coupled_block_share_uncontrolled'])}. Table~\\ref{{tab:gci-overview}} reports all three thresholds and Figure~\\ref{{fig:gci}} the
distances, the nested model sets, and the per-agent shares.

\\textbf{{Cross-vendor effect.}} Restricting the alternatives to {vend(a.models[0])} models gives a coupled share of
{cvc.get('same_vendor_as_first_model_only', float('nan')):.2f} for controlled and {cvu.get('same_vendor_as_first_model_only', float('nan')):.2f} for swap-only
blocks; adding the other vendor's model gives {cvc.get('all_models', float('nan')):.2f} and {cvu.get('all_models', float('nan')):.2f}. As Theorem~\\ref{{thm:onesided}}
requires, the estimate never increases as models are added; the size of the drop measures how much the second vendor changes the verdict.

\\textbf{{Robustness.}} At $\\theta = {th0:g}$, $F(0.05) = {m0['F_delta']['0.05']:.2f}$: a perturbation of the alternative-model distances by
$0.05$ changes GCI by at most this amount (Theorem~\\ref{{thm:gcirob}}).

\\begin{{table*}}[t]\\caption{{Generative coupling: shares of blocks (95\\% cluster-bootstrap intervals over repositories)}}\\label{{tab:gci-overview}}
\\centering\\footnotesize\\input{{tab_gci_overview.tex}}\\end{{table*}}
\\begin{{figure}}[t]\\centering
\\includegraphics[width=\\linewidth]{{figs/fig_gci_nested.pdf}}\\\\[2pt]
\\includegraphics[width=\\linewidth]{{figs/fig_gci_agents.pdf}}
\\caption{{Generative coupling. Top: share of blocks coupled as alternative models are added (Theorem~\\ref{{thm:onesided}}). Bottom: coupled
share by agent at $\\theta = {th0:g}$; blue bars passed the self-regeneration control, grey bars are swap-only.}}\\label{{fig:gci}}\\end{{figure}}
"""
    (out / "section_gci_draft.tex").write_text(tex)

    print(json.dumps({k: R[k] for k in ["n_blocks", "n_repos", "controlled_blocks", "theta_main"]}, indent=2))
    m0 = R["by_theta"][str(th0)]
    print(f"theta={th0:g}: under-specified {fmt_ci(m0['under_specified_share_controlled'])}; "
          f"coupled (controlled) {fmt_ci(m0['coupled_block_share_controlled'])}; "
          f"coupled (swap only) {fmt_ci(m0['coupled_block_share_uncontrolled'])}")
    print("nested (controlled):", m0["coupled_share_nested"]["controlled"], "| within vs cross vendor:", m0["within_vs_cross_vendor"])
    print("wrote tables and figures to", out)


if __name__ == "__main__":
    main()
