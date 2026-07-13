PYTHON ?= python3
PREPARATION_PYTHON ?= .venv/bin/python
PREPARATION_BOOTSTRAP_PYTHON ?= python3.12

.PHONY: help check new-experiment prepare-environment prepare-inputs

help:
	@echo "make check"
	@echo "make new-experiment ID=EXP-NN SLUG=short_slug TITLE='Experiment title'"
	@echo "make prepare-environment"
	@echo "make prepare-inputs"

check:
	$(PYTHON) -m py_compile scripts/new_experiment.py scripts/prepare_inputs.py scripts/validate_repo.py experiments/_template/analyze.py
	$(PYTHON) scripts/new_experiment.py --help
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) scripts/validate_repo.py
	git diff --check

new-experiment:
	@test -n "$(ID)" || (echo "ID is required, for example ID=EXP-01"; exit 2)
	@test -n "$(SLUG)" || (echo "SLUG is required"; exit 2)
	@test -n "$(TITLE)" || (echo "TITLE is required"; exit 2)
	$(PYTHON) scripts/new_experiment.py "$(ID)" "$(SLUG)" "$(TITLE)"

prepare-environment:
	test -x "$(PREPARATION_PYTHON)" || $(PREPARATION_BOOTSTRAP_PYTHON) -m venv .venv
	$(PREPARATION_PYTHON) -m pip install -r requirements-preparation.txt

prepare-inputs:
	test -x "$(PREPARATION_PYTHON)" || (echo "run 'make prepare-environment' first"; exit 2)
	$(PREPARATION_PYTHON) scripts/prepare_inputs.py --config configs/inputs.lock.yaml
