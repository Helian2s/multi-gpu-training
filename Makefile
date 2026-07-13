PYTHON ?= python3

.PHONY: help check new-experiment

help:
	@echo "make check"
	@echo "make new-experiment ID=EXP-NN SLUG=short_slug TITLE='Experiment title'"

check:
	$(PYTHON) -m py_compile scripts/new_experiment.py scripts/validate_repo.py experiments/_template/analyze.py
	$(PYTHON) scripts/new_experiment.py --help
	$(PYTHON) scripts/validate_repo.py
	git diff --check

new-experiment:
	@test -n "$(ID)" || (echo "ID is required, for example ID=EXP-01"; exit 2)
	@test -n "$(SLUG)" || (echo "SLUG is required"; exit 2)
	@test -n "$(TITLE)" || (echo "TITLE is required"; exit 2)
	$(PYTHON) scripts/new_experiment.py "$(ID)" "$(SLUG)" "$(TITLE)"
