PYTHON ?= python3
PREPARATION_PYTHON ?= .venv/bin/python
PREPARATION_BOOTSTRAP_PYTHON ?= python3.12
IMAGE_PLATFORM ?= linux/amd64
PYTORCH_IMAGE ?= multi-gpu-training-pytorch:local
NEMO_IMAGE ?= multi-gpu-training-nemo:local
VCS_REF ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
AWS_EXP01_CONFIG ?= infra/aws/exp01_qualification.yaml
AWS_EXP01_INSTANCE_ARG = $(if $(INSTANCE_ID),--instance-id $(INSTANCE_ID),)
AWS_EXP01_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
AWS_EXP01_HOLD_ARG = $(if $(HOLD_OPEN_ON_EXIT),--hold-open-on-exit,)

.PHONY: help check new-experiment prepare-environment prepare-inputs verify-inputs build-pytorch-image build-nemo-image aws-ssm-plugin-check aws-exp01-preflight aws-exp01-launch-dry-run aws-exp01-status aws-exp01-host-shell aws-exp01-container-shell aws-exp01-host-command aws-exp01-container-command aws-exp01-logs aws-exp01-monitor aws-exp01-artifacts

help:
	@echo "make check"
	@echo "make build-pytorch-image [PYTORCH_IMAGE=multi-gpu-training-pytorch:local]"
	@echo "make build-nemo-image [NEMO_IMAGE=multi-gpu-training-nemo:local]"
	@echo "make aws-ssm-plugin-check"
	@echo "make aws-exp01-preflight"
	@echo "make aws-exp01-launch-dry-run [RUN_ID=...] [HOLD_OPEN_ON_EXIT=1]"
	@echo "make aws-exp01-status"
	@echo "make aws-exp01-host-shell [INSTANCE_ID=...]"
	@echo "make aws-exp01-container-shell [INSTANCE_ID=...] [RUN_ID=...]"
	@echo "make aws-exp01-host-command CMD='...' [INSTANCE_ID=...]"
	@echo "make aws-exp01-container-command CMD='...' [INSTANCE_ID=...] [RUN_ID=...]"
	@echo "make aws-exp01-logs [INSTANCE_ID=...]"
	@echo "make aws-exp01-monitor [INSTANCE_ID=...]"
	@echo "make aws-exp01-artifacts"
	@echo "make new-experiment ID=EXP-NN SLUG=short_slug TITLE='Experiment title'"
	@echo "make prepare-environment"
	@echo "make prepare-inputs"
	@echo "make verify-inputs"

check:
	$(PYTHON) -m py_compile scripts/new_experiment.py scripts/prepare_inputs.py scripts/validate_repo.py infra/aws/exp01_preflight.py infra/aws/exp01_launch.py infra/aws/exp01_ops.py experiments/_template/analyze.py experiments/exp_01_aws_pcie_p2p_nccl_communication/analyze.py
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
	$(PYTHON) infra/aws/exp01_preflight.py --config "$(AWS_EXP01_CONFIG)"

aws-ssm-plugin-check:
	@command -v session-manager-plugin >/dev/null 2>&1 && session-manager-plugin --version || (echo "session-manager-plugin is not installed; interactive SSM shell targets need it"; exit 2)

aws-exp01-launch-dry-run:
	$(PYTHON) infra/aws/exp01_launch.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_RUN_ARG) $(AWS_EXP01_HOLD_ARG)

aws-exp01-status:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" status

aws-exp01-host-shell:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_INSTANCE_ARG) shell

aws-exp01-container-shell:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_INSTANCE_ARG) $(AWS_EXP01_RUN_ARG) container-shell

aws-exp01-host-command:
	@test -n "$(CMD)" || (echo "CMD is required, for example CMD='nvidia-smi'"; exit 2)
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_INSTANCE_ARG) command --shell-command '$(CMD)'

aws-exp01-container-command:
	@test -n "$(CMD)" || (echo "CMD is required, for example CMD='python -c \"import torch; print(torch.cuda.device_count())\"'"; exit 2)
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_INSTANCE_ARG) $(AWS_EXP01_RUN_ARG) container-command --shell-command '$(CMD)'

aws-exp01-logs:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_INSTANCE_ARG) logs

aws-exp01-monitor:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_INSTANCE_ARG) monitor

aws-exp01-artifacts:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_EXP01_CONFIG)" artifacts
