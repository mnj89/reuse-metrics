"""Builds and executes notebooks/worked_example.ipynb. Run from the repository root."""
import os
from pathlib import Path
import nbformat as nbf

# the notebook kernel may be a different interpreter: make the library importable without an install
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Reproducing the illustrative computation (Section VII)

This notebook recomputes every number in the worked example of
*Measuring Software Reuse When the Reusing Agent is a Foundation Model*
using the `reuse_metrics` reference implementation. The provenance of each
unit is **stipulated by construction**, exactly as in the paper.""")

code("""import numpy as np
from reuse_metrics import (Attribution, Event, Prov, System, Unit, component_vector, hrs,
                           information_loss, lrd_typicality_expectation, profile, report)

# Table II: 200 logical lines
spec = [(Prov.D, 30), (Prov.P, 30), (Prov.L, 80), (Prov.G, 40), (Prov.H, 20)]
units, attrs, truth, i = [], {}, {}, 0
for c, n in spec:
    for _ in range(n):
        u = Unit("bookmarks.py", i, "r1", f"line {i}")
        units.append(u); truth[u.key] = c
        if c == Prov.D:  # 20 lines copied from the Flask quickstart (attributed) + 10 imports
            attrs[u.key] = Attribution("flask", "copy" if i < 20 else "import")
        i += 1
keys = {c: [u.key for u in units if truth[u.key] == c] for c in Prov}

# Five generation events; one names a pattern in N
events = [
    Event("e1", 1, "a SQLAlchemy model for bookmarks", "m", frozenset(keys[Prov.P])),
    Event("e2", 2, "add CRUD routes", "m", frozenset(keys[Prov.L][:40])),
    Event("e3", 3, "add CRUD routes for tags", "m", frozenset(keys[Prov.L][40:])),
    Event("e4", 4, "write a tag-cloud algorithm", "m", frozenset(keys[Prov.G])),
    Event("e5", 5, "tweak the response format", "m", frozenset()),
]
system = System(units, attrs, events)
N = {"SQLAlchemy"}
reconstructed = {"e1", "e2", "e3", "e4"}   # 4 of 5 stored prompts replay successfully
beta = (0.30, 0.20, 0.15, 0.15, 0.20)""")

md("## The six metrics and HRS")
code("""r = report(truth, system.events, N, reconstructed, beta)
for k in ["DRR", "LRD", "PPC", "PPC_U", "PLC", "PRF", "PPA", "GCI", "HRS"]:
    print(f"{k:6s} = {r[k]:.3f}")""")

md("## What the typicality estimator would report (Eq. 4)")
code("""p = profile(truth)
print("E[LRD_typ] =", round(lrd_typicality_expectation(p, {Prov.L: .9, Prov.P: .9, Prov.G: .9}, fp_human=.2), 3))""")

md("## Information loss of a deliberate-only measure (Theorem 4)")
code("""loss, ceiling = information_loss(p)
print(f"L(U) = {loss:.3f} bits/unit   ceiling 2(1-p_D) = {ceiling:.2f}")
all_human = {Prov.D: .15, Prov.L: 0, Prov.P: 0, Prov.G: 0, Prov.H: .85}
print("all-human alternative:", information_loss(all_human)[0])""")

md("## Sensitivity table (Table III)")
code("""lo = component_vector(0.14, 0.48, 1/6, 0.60, 0.23)
hi = component_vector(0.16, 0.32, 0.25, 1.00, 0.18)
print(f"HRS worst-case range: [{hrs(lo, beta):.4f}, {hrs(hi, beta):.4f}]  -> reported [0.41, 0.54]")
print("PPC with one non-pattern event removed/added:", {round(1/4,2), round(1/5,2), round(1/6,2)})
print("PPA with one replay more/fewer:", {3/5, 4/5, 5/5})""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ExecutePreprocessor(timeout=120, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "notebooks/worked_example.ipynb")
for c in nb.cells:
    for o in c.get("outputs", []):
        if o.get("text"): print(o["text"])
