# Reproduction targets. Run `make help`.
PY ?= python

help:
	@echo "make install      install the library and the analysis dependencies"
	@echo "make test         run the test suite (worked example and formal results)"
	@echo "make example      run the synthetic end-to-end demo"
	@echo "make notebook     rebuild and execute the worked-example notebook"
	@echo "make field-study  recompute every number of the field study from the stored runs"
	@echo "make pilot        re-score the swap-test pilot from the stored model answers (no API calls)"
	@echo "make figures      rebuild all five figures (needs a LaTeX installation)"
	@echo "make all          test, field-study, pilot"

install:
	$(PY) -m pip install -e ".[dev,studies]"

test:
	$(PY) -m pytest -q

example:
	$(PY) examples/run_synthetic.py --seed 3

notebook:
	$(PY) notebooks/build_notebook.py

field-study:
	$(PY) studies/field_study/analyse.py

pilot:
	$(PY) studies/swap_test_pilot/rescore_pilot.py

figures:
	cd figures && for f in fig1_framework fig2_taxonomy fig4_example; do pdflatex -interaction=nonstopmode $$f.tex > /dev/null && echo "built $$f.pdf"; done
	cd figures && $(PY) figs_data.py

all: test field-study pilot

.PHONY: help install test example notebook field-study pilot figures all
