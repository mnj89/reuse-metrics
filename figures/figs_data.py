"""Data figures at their printed size, so that font sizes in the PDF are the sizes on the page.
Fig. 2 (disruption) and Fig. 5 (field study) are both 6.8 in wide (0.95 of the IEEE text width)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Text is typeset by LaTeX (pgf backend) with the same fonts as the TikZ figures: Type 1, fully embedded.
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": True,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{sansmath}\renewcommand{\familydefault}{\sfdefault}\sansmath",
    "font.family": "sans-serif",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
cD, cL, cP, cG, cH, INK, RULE = "#1F5FAE", "#1B7F4B", "#C2590F", "#6A3D9A", "#4D4D4D", "#1A1A1A", "#8C8C8C"
OUT = "./"   # run from this folder
W = 6.8

# ------------------------------------------------------------------ Fig. 2: disruption
fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 2.55), gridspec_kw={"width_ratios": [0.8, 1.05, 1.3]}, layout="constrained")
years, share = ["2023", "2024", "2025"], [44, 62, 79]
a.bar(years, share, color=["#A9C3E6", "#5F8FD0", cD], width=0.62)
for x, v in enumerate(share):
    a.text(x, v + 2, f"{v}\\%", ha="center", va="bottom", fontsize=8.5, color=INK)
a.set_ylim(0, 100); a.set_yticks([0, 25, 50, 75, 100])
a.set_ylabel(r"Developers using AI tools (\%)"); a.set_title(r"\textbf{AI tool adoption}")

yrs, churn, proj = [2020, 2021, 2022, 2023], [3.32, 3.63, 3.97, 5.53], 7.09
b.plot(yrs, churn, "-o", color=cP, lw=1.6, ms=5)
b.plot([2023, 2024], [5.53, proj], "--", color=cP, lw=1.2)
b.plot([2024], [proj], "o", mfc="white", mec=cP, ms=5.5, mew=1.3)
for x, v in zip(yrs, churn):
    b.text(x, v + 0.45, f"{v:.1f}\\%", ha="center", fontsize=8, color=INK)
b.text(2024, proj + 0.45, f"{proj:.1f}\\%*", ha="center", fontsize=8, color=INK)
b.set_xticks(yrs + [2024]); b.set_xlim(2019.6, 2024.5); b.set_ylim(0, 9); b.set_yticks([0, 3, 6, 9])
b.set_ylabel(r"Lines changed within 2 weeks (\%)"); b.set_title(r"\textbf{Two-week code churn}")
b.text(0.03, 0.04, r"\textit{* projection in the source}", transform=b.transAxes, fontsize=7.5, color=cH)

ops = ["Updated", "Copy/pasted", "Added", "Deleted", "Moved"]
v20 = [5.19, 8.26, 39.18, 19.47, 24.99]; v23 = [5.50, 10.49, 42.34, 21.12, 16.92]
for y, (s, e, name) in enumerate(zip(v20, v23, ops)):
    col = cG if name in ("Copy/pasted", "Moved") else RULE
    c.plot([s, e], [y, y], color=col, lw=1.8, zorder=1)
    c.plot([s], [y], "o", color="white", mec=col, ms=5.5, mew=1.3, zorder=2)
    c.plot([e], [y], "o", color=col, ms=5.5, zorder=2)
    c.text(max(s, e) + 2.0, y, f"{s:.1f} → {e:.1f}", va="center", fontsize=8,
           color=cG if col == cG else INK, fontweight="normal")
c.set_yticks(range(len(ops))); c.set_yticklabels(ops); c.set_xlim(0, 64); c.set_ylim(-0.6, 4.6)
c.set_xticks([0, 20, 40, 60]); c.set_xlabel(r"Share of changed lines (\%)")
c.set_title(r"\textbf{Share of changed lines by operation}")
c.text(0.98, 0.03, r"\textit{open: 2020, filled: 2023}", transform=c.transAxes, ha="right", fontsize=7.5, color=cH)
fig.savefig(OUT + "fig3_disruption.pdf", backend="pgf")
plt.close(fig)

# ------------------------------------------------------------------ Fig. 5: field study
p = pd.read_csv("../results/field_study/pooled_per_repo.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw={"width_ratios": [1.12, 1]}, layout="constrained")
cols = ["DRR", "LRD_log", "PPC", "PPC_U", "HRS_equal_weights"]
labels = ["DRR", "LRD\n(log-based)", "PPC", r"PPC$_U$", "HRS\n(equal weights)"]
colors = [cD, cL, cP, cP, INK]
rng = np.random.default_rng(7)
for i, (col, colr) in enumerate(zip(cols, colors), start=1):
    v = p[col].values
    a.scatter(i - 0.26 + rng.uniform(-0.09, 0.09, len(v)), v, s=5, color=colr, alpha=0.35, linewidths=0, zorder=1)
    bp = a.boxplot([v], positions=[i + 0.10], widths=0.30, patch_artist=True, showfliers=False, zorder=2,
                   medianprops=dict(color="white", lw=1.4), boxprops=dict(facecolor=colr, edgecolor=colr, lw=0.8),
                   whiskerprops=dict(color=colr, lw=0.9), capprops=dict(color=colr, lw=0.9))
    med = float(np.median(v))
    a.text(i + 0.31, med, r"\textbf{%.2f}" % med, va="center", ha="left", fontsize=8, color=colr)
a.set_xticks(range(1, 6)); a.set_xticklabels(labels); a.set_xlim(0.45, 5.85)
a.set_ylim(-0.03, 1.03); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a.set_ylabel("Value per repository")
a.set_title(r"\textbf{Metric distributions over %d repositories}" % len(p))

sc = b.scatter(p.DRR, p.LRD_log, c=p.PPC, cmap="viridis", vmin=0, vmax=0.75, s=20, edgecolor="white", linewidth=0.4)
b.set_xlabel("DRR"); b.set_ylabel("LRD (log-based)"); b.set_title(r"\textbf{Equal DRR, different profiles}")
b.set_xlim(-0.006, 0.23); b.set_ylim(-0.03, 1.03); b.set_xticks([0, 0.05, 0.10, 0.15, 0.20]); b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
cb = fig.colorbar(sc, ax=b, pad=0.02, fraction=0.05); cb.set_label("PPC"); cb.set_ticks([0, 0.25, 0.5, 0.75]); cb.outline.set_linewidth(0.5)
fig.savefig(OUT + "fig5_studya.pdf", backend="pgf")
print("ok", len(p), [round(float(p[c].median()), 3) for c in cols])
