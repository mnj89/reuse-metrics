# Figure sources

| Paper | Source | Build |
|---|---|---|
| Fig. 1 (framework) | `fig1_framework.tex` | `pdflatex fig1_framework.tex` |
| Fig. 3 (classes) | `fig2_taxonomy.tex` | `pdflatex fig2_taxonomy.tex` |
| Fig. 4 (worked example) | `fig4_example.tex` | `pdflatex fig4_example.tex` |
| Fig. 2 (trends), Fig. 5 (field study) | `figs_data.py` | `python figs_data.py` (run from this folder) |

`figstyle.tex` holds the shared colours and styles. `figs_data.py` uses matplotlib's pgf backend, so it needs a LaTeX
installation; it writes `fig3_disruption.pdf` (Fig. 2) and `fig5_studya.pdf` (Fig. 5). Or run `make figures` from the root.
