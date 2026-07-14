PYTHON ?= python3
PREPARATION_PYTHON ?= .venv/bin/python
PREPARATION_BOOTSTRAP_PYTHON ?= python3.12
IMAGE_PLATFORM ?= linux/amd64
PYTORCH_IMAGE ?= multi-gpu-training-pytorch:local
NEMO_IMAGE ?= multi-gpu-training-nemo:local
VCS_REF ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: help check new-experiment prepare-environment prepare-inputs verify-inputs build-pytorch-image build-nemo-image aws-exp01-preflight

help:
	@echo "make check"
	@echo "make build-pytorch-image [PYTORCH_IMAGE=multi-gpu-training-pytorch:local]"
	@echo "make build-nemo-image [NEMO_IMAGE=multi-gpu-training-nemo:local]"
	@echo "make aws-exp01-preflight"
	@echo "make new-experiment ID=EXP-NN SLUG=short_slug TITLE='Experiment title'"
	@echo "make prepare-environment"
	@echo "make prepare-inputs"
	@echo "make verify-inputs"

check:
	$(PYTHON) -m py_compile scripts/new_experiment.py scripts/prepare_inputs.py scripts/validate_repo.py infra/aws/exp01_preflight.py experiments/_template/analyze.py experiments/exp_01_aws_pcie_p2p_nccl_communication/analyze.py
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

verify-inputs:
	test -x "$(PREPARATION_PYTHON)" || (echo "run 'make prepare-environment' first"; exit 2)
	$(PREPARATION_PYTHON) scripts/prepare_inputs.py --config configs/inputs.lock.yaml --verify-only

build-pytorch-image:
	docker buildx build --platform "$(IMAGE_PLATFORM)" --load \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		-f containers/pytorch/Dockerfile \
		-t "$(PYTORCH_IMAGE)" .

build-nemo-image:
	docker buildx build --platform "$(IMAGE_PLATFORM)" --load \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		-f containers/nemo/Dockerfile \
		-t "$(NEMO_IMAGE)" .

aws-exp01-preflight:
	$(PYTHON) infra/aws/exp01_preflight.py --config infra/aws/exp01_qualification.yaml
