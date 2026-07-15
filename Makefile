PYTHON ?= python3
PREPARATION_PYTHON ?= .venv/bin/python
PREPARATION_BOOTSTRAP_PYTHON ?= python3.12
IMAGE_PLATFORM ?= linux/amd64
PYTORCH_IMAGE ?= multi-gpu-training-pytorch:local
NEMO_IMAGE ?= multi-gpu-training-nemo:local
NEMO_BASE_IMAGE ?= nvcr.io/nvidia/nemo:26.06@sha256:bb1dbe94646d5a6490570823cafa0d6f753e1cb60df8f5e89e3b32f3f87893fc
CUDA_SAMPLES_ARCHITECTURES ?= 80
VCS_REF ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
AWS_EXP01_CONFIG ?= infra/aws/exp01_qualification.yaml
AWS_A1_CONFIG ?= infra/aws/a1_qualification.yaml
AWS_A1_QUEUE_CONFIG ?= infra/aws/a1_experiment_queue.yaml
AWS_A2_QUEUE_CONFIG ?= infra/aws/a2_experiment_queue.yaml
AWS_A2_MEGATRON_QUEUE_CONFIG ?= infra/aws/a2_megatron_queue.yaml
RUNPOD_A2_MEGATRON_QUEUE_CONFIG ?= infra/runpod/a2_megatron_queue.yaml
RUNPOD_A4_MEGATRON_QUEUE_CONFIG ?= infra/runpod/a4_megatron_queue.yaml
AWS_EXP01_INSTANCE_ARG = $(if $(INSTANCE_ID),--instance-id $(INSTANCE_ID),)
AWS_EXP01_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
AWS_EXP01_HOLD_ARG = $(if $(HOLD_OPEN_ON_EXIT),--hold-open-on-exit,)
AWS_A1_INSTANCE_ARG = $(if $(INSTANCE_ID),--instance-id $(INSTANCE_ID),)
AWS_A1_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
AWS_A1_HOLD_ARG = $(if $(HOLD_OPEN_ON_EXIT),--hold-open-on-exit,)
AWS_A1_QUEUE_INSTANCE_ARG = $(if $(INSTANCE_ID),--instance-id $(INSTANCE_ID),)
AWS_A1_QUEUE_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
AWS_A2_QUEUE_INSTANCE_ARG = $(if $(INSTANCE_ID),--instance-id $(INSTANCE_ID),)
AWS_A2_QUEUE_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
AWS_A2_QUEUE_CONFIRM_ARG = $(if $(CONFIRM),--confirm "$(CONFIRM)",)
AWS_A2_MEGATRON_QUEUE_INSTANCE_ARG = $(if $(INSTANCE_ID),--instance-id $(INSTANCE_ID),)
AWS_A2_MEGATRON_QUEUE_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
AWS_A2_MEGATRON_QUEUE_CONFIRM_ARG = $(if $(CONFIRM),--confirm "$(CONFIRM)",)
RUNPOD_A2_MEGATRON_QUEUE_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)
RUNPOD_A4_MEGATRON_QUEUE_RUN_ARG = $(if $(RUN_ID),--run-id $(RUN_ID),)

.PHONY: help check new-experiment prepare-environment prepare-inputs verify-inputs exp-a1-dry-run exp-a2-dry-run exp-a2-megatron-dry-run exp-runpod-megatron-dry-run build-pytorch-image build-nemo-image aws-ssm-plugin-check aws-exp01-preflight aws-exp01-launch-dry-run aws-exp01-status aws-exp01-host-shell aws-exp01-container-shell aws-exp01-host-command aws-exp01-container-command aws-exp01-logs aws-exp01-monitor aws-exp01-artifacts aws-a1-preflight aws-a1-launch-dry-run aws-a1-status aws-a1-host-shell aws-a1-container-shell aws-a1-host-command aws-a1-container-command aws-a1-logs aws-a1-monitor aws-a1-artifacts aws-a1-queue-plan aws-a1-queue-script aws-a1-queue-run aws-a2-queue-plan aws-a2-queue-script aws-a2-queue-launch-dry-run aws-a2-queue-launch aws-a2-queue-run aws-a2-megatron-queue-plan aws-a2-megatron-queue-script aws-a2-megatron-queue-launch-dry-run aws-a2-megatron-queue-launch aws-a2-megatron-queue-run runpod-a2-megatron-queue-plan runpod-a2-megatron-queue-script runpod-a2-megatron-pod-create-command runpod-a4-megatron-queue-plan runpod-a4-megatron-queue-script runpod-a4-megatron-pod-create-command

help:
	@echo "make check"
	@echo "make build-pytorch-image [PYTORCH_IMAGE=multi-gpu-training-pytorch:local]"
	@echo "make build-nemo-image [NEMO_IMAGE=multi-gpu-training-nemo:local]"
	@echo "make aws-ssm-plugin-check"
	@echo "make aws-a1-preflight"
	@echo "make aws-a1-launch-dry-run [RUN_ID=...] [HOLD_OPEN_ON_EXIT=1]"
	@echo "make aws-a1-status"
	@echo "make aws-a1-host-shell [INSTANCE_ID=...]"
	@echo "make aws-a1-container-shell [INSTANCE_ID=...] [RUN_ID=...]"
	@echo "make aws-a1-host-command CMD='...' [INSTANCE_ID=...]"
	@echo "make aws-a1-container-command CMD='...' [INSTANCE_ID=...] [RUN_ID=...]"
	@echo "make aws-a1-logs [INSTANCE_ID=...]"
	@echo "make aws-a1-monitor [INSTANCE_ID=...]"
	@echo "make aws-a1-artifacts"
	@echo "make aws-a1-queue-plan"
	@echo "make aws-a1-queue-script [RUN_ID=...]"
	@echo "make aws-a1-queue-run INSTANCE_ID=i-... [RUN_ID=...]"
	@echo "make aws-a2-queue-plan"
	@echo "make aws-a2-queue-script [RUN_ID=...]"
	@echo "make aws-a2-queue-launch-dry-run [RUN_ID=...]"
	@echo "make aws-a2-queue-launch RUN_ID=... CONFIRM='launch AWS-A2-PyTorch AWS-A2 stop-after-300m'"
	@echo "make aws-a2-queue-run INSTANCE_ID=i-... [RUN_ID=...]"
	@echo "make aws-a2-megatron-queue-plan"
	@echo "make aws-a2-megatron-queue-script [RUN_ID=...]"
	@echo "make aws-a2-megatron-queue-launch-dry-run [RUN_ID=...]"
	@echo "make aws-a2-megatron-queue-launch RUN_ID=... CONFIRM='launch AWS-A2-Megatron AWS-A2 stop-after-300m'"
	@echo "make aws-a2-megatron-queue-run INSTANCE_ID=i-... [RUN_ID=...]"
	@echo "make runpod-a2-megatron-queue-plan"
	@echo "make runpod-a2-megatron-queue-script [RUN_ID=...]"
	@echo "make runpod-a2-megatron-pod-create-command RUN_ID=..."
	@echo "make runpod-a4-megatron-queue-plan"
	@echo "make runpod-a4-megatron-queue-script [RUN_ID=...]"
	@echo "make runpod-a4-megatron-pod-create-command RUN_ID=..."
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
	@echo "make exp-a1-dry-run"
	@echo "make exp-a2-dry-run"
	@echo "make exp-a2-megatron-dry-run"
	@echo "make exp-runpod-megatron-dry-run"

check:
	$(PYTHON) -m py_compile scripts/new_experiment.py scripts/prepare_inputs.py scripts/validate_repo.py infra/aws/exp01_preflight.py infra/aws/exp01_launch.py infra/aws/exp01_ops.py infra/aws/a1_queue.py infra/aws/a2_queue.py infra/runpod/runpod_queue.py common/experiment_runner.py common/pytorch_executor.py common/megatron_executor.py common/qualification/aws_gpu_smoke.py common/qualification/runpod_gpu_smoke.py experiments/_template/analyze.py experiments/exp_*/analyze.py experiments/exp_*/run_exp*.py
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

exp-a1-dry-run:
	$(PYTHON) experiments/exp_03_microbatch_gradient_accumulation/run_exp03.py --dry-run
	$(PYTHON) experiments/exp_04_activation_checkpointing_recomputation/run_exp04.py --dry-run
	$(PYTHON) experiments/exp_05_sdpa_flashattention_operator_fusion/run_exp05.py --dry-run
	$(PYTHON) experiments/exp_06_profiler_triangulation/run_exp06.py --dry-run

exp-a2-dry-run:
	$(PYTHON) experiments/exp_02_mixed_precision_tensor_cores/run_exp02.py --dry-run
	$(PYTHON) experiments/exp_07_ddp_scaling_communication_overlap/run_exp07.py --dry-run
	$(PYTHON) experiments/exp_08_fsdp_sharding_zero_memory_tradeoffs/run_exp08.py --dry-run
	$(PYTHON) experiments/exp_09_controlled_troubleshooting_failure_diagnosis/run_exp09.py --dry-run

exp-a2-megatron-dry-run:
	$(PYTHON) experiments/exp_12_pipeline_schedules_bubble_size/run_exp12.py --dry-run --allow-unpublished-image

exp-runpod-megatron-dry-run:
	$(PYTHON) experiments/exp_11_tensor_sequence_parallelism/run_exp11.py --dry-run --allow-unpublished-image
	$(PYTHON) experiments/exp_13_context_parallelism_long_sequences/run_exp13.py --dry-run --allow-unpublished-image
	$(PYTHON) experiments/exp_14_tp2_dp2_hybrid/run_exp14.py --dry-run --allow-unpublished-image

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
		--build-arg NEMO_BASE_IMAGE="$(NEMO_BASE_IMAGE)" \
		--build-arg CUDA_SAMPLES_ARCHITECTURES="$(CUDA_SAMPLES_ARCHITECTURES)" \
		-f containers/nemo/Dockerfile \
		-t "$(NEMO_IMAGE)" .

aws-exp01-preflight:
	$(PYTHON) infra/aws/exp01_preflight.py --config "$(AWS_EXP01_CONFIG)"

aws-a1-preflight:
	$(PYTHON) infra/aws/exp01_preflight.py --config "$(AWS_A1_CONFIG)"

aws-ssm-plugin-check:
	@command -v session-manager-plugin >/dev/null 2>&1 && session-manager-plugin --version || (echo "session-manager-plugin is not installed; interactive SSM shell targets need it"; exit 2)

aws-exp01-launch-dry-run:
	$(PYTHON) infra/aws/exp01_launch.py --config "$(AWS_EXP01_CONFIG)" $(AWS_EXP01_RUN_ARG) $(AWS_EXP01_HOLD_ARG)

aws-a1-launch-dry-run:
	$(PYTHON) infra/aws/exp01_launch.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_RUN_ARG) $(AWS_A1_HOLD_ARG)

aws-a1-status:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" status

aws-a1-host-shell:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_INSTANCE_ARG) shell

aws-a1-container-shell:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_INSTANCE_ARG) $(AWS_A1_RUN_ARG) container-shell

aws-a1-host-command:
	@test -n "$(CMD)" || (echo "CMD is required, for example CMD='nvidia-smi'"; exit 2)
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_INSTANCE_ARG) command --shell-command '$(CMD)'

aws-a1-container-command:
	@test -n "$(CMD)" || (echo "CMD is required, for example CMD='python -c \"import torch; print(torch.cuda.device_count())\"'"; exit 2)
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_INSTANCE_ARG) $(AWS_A1_RUN_ARG) container-command --shell-command '$(CMD)'

aws-a1-logs:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_INSTANCE_ARG) logs

aws-a1-monitor:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" $(AWS_A1_INSTANCE_ARG) monitor

aws-a1-artifacts:
	$(PYTHON) infra/aws/exp01_ops.py --config "$(AWS_A1_CONFIG)" artifacts

aws-a1-queue-plan:
	$(PYTHON) infra/aws/a1_queue.py --queue-config "$(AWS_A1_QUEUE_CONFIG)" plan

aws-a1-queue-script:
	$(PYTHON) infra/aws/a1_queue.py --queue-config "$(AWS_A1_QUEUE_CONFIG)" $(AWS_A1_QUEUE_RUN_ARG) host-script

aws-a1-queue-run:
	@test -n "$(INSTANCE_ID)" || (echo "INSTANCE_ID is required, for example INSTANCE_ID=i-..."; exit 2)
	$(PYTHON) infra/aws/a1_queue.py --queue-config "$(AWS_A1_QUEUE_CONFIG)" $(AWS_A1_QUEUE_INSTANCE_ARG) $(AWS_A1_QUEUE_RUN_ARG) run

aws-a2-queue-plan:
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_QUEUE_CONFIG)" plan

aws-a2-queue-script:
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_QUEUE_CONFIG)" $(AWS_A2_QUEUE_RUN_ARG) host-script

aws-a2-queue-launch-dry-run:
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_QUEUE_CONFIG)" $(AWS_A2_QUEUE_RUN_ARG) launch-dry-run

aws-a2-queue-launch:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required, for example RUN_ID=aws-a2-pytorch-$$(date -u +%Y%m%dT%H%M%SZ)"; exit 2)
	@test -n "$(CONFIRM)" || (echo "CONFIRM is required: launch AWS-A2-PyTorch AWS-A2 stop-after-300m"; exit 2)
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_QUEUE_CONFIG)" $(AWS_A2_QUEUE_RUN_ARG) launch $(AWS_A2_QUEUE_CONFIRM_ARG)

aws-a2-queue-run:
	@test -n "$(INSTANCE_ID)" || (echo "INSTANCE_ID is required, for example INSTANCE_ID=i-..."; exit 2)
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_QUEUE_CONFIG)" $(AWS_A2_QUEUE_INSTANCE_ARG) $(AWS_A2_QUEUE_RUN_ARG) run

aws-a2-megatron-queue-plan:
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_MEGATRON_QUEUE_CONFIG)" plan

aws-a2-megatron-queue-script:
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_MEGATRON_QUEUE_CONFIG)" $(AWS_A2_MEGATRON_QUEUE_RUN_ARG) host-script

aws-a2-megatron-queue-launch-dry-run:
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_MEGATRON_QUEUE_CONFIG)" $(AWS_A2_MEGATRON_QUEUE_RUN_ARG) launch-dry-run

aws-a2-megatron-queue-launch:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required, for example RUN_ID=aws-a2-megatron-$$(date -u +%Y%m%dT%H%M%SZ)"; exit 2)
	@test -n "$(CONFIRM)" || (echo "CONFIRM is required: launch AWS-A2-Megatron AWS-A2 stop-after-300m"; exit 2)
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_MEGATRON_QUEUE_CONFIG)" $(AWS_A2_MEGATRON_QUEUE_RUN_ARG) launch $(AWS_A2_MEGATRON_QUEUE_CONFIRM_ARG)

aws-a2-megatron-queue-run:
	@test -n "$(INSTANCE_ID)" || (echo "INSTANCE_ID is required, for example INSTANCE_ID=i-..."; exit 2)
	$(PYTHON) infra/aws/a2_queue.py --queue-config "$(AWS_A2_MEGATRON_QUEUE_CONFIG)" $(AWS_A2_MEGATRON_QUEUE_INSTANCE_ARG) $(AWS_A2_MEGATRON_QUEUE_RUN_ARG) run

runpod-a2-megatron-queue-plan:
	$(PYTHON) infra/runpod/runpod_queue.py --queue-config "$(RUNPOD_A2_MEGATRON_QUEUE_CONFIG)" plan

runpod-a2-megatron-queue-script:
	$(PYTHON) infra/runpod/runpod_queue.py --queue-config "$(RUNPOD_A2_MEGATRON_QUEUE_CONFIG)" $(RUNPOD_A2_MEGATRON_QUEUE_RUN_ARG) container-script

runpod-a2-megatron-pod-create-command:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required, for example RUN_ID=runpod-a2-megatron-$$(date -u +%Y%m%dT%H%M%SZ)"; exit 2)
	$(PYTHON) infra/runpod/runpod_queue.py --queue-config "$(RUNPOD_A2_MEGATRON_QUEUE_CONFIG)" $(RUNPOD_A2_MEGATRON_QUEUE_RUN_ARG) pod-create-command

runpod-a4-megatron-queue-plan:
	$(PYTHON) infra/runpod/runpod_queue.py --queue-config "$(RUNPOD_A4_MEGATRON_QUEUE_CONFIG)" plan

runpod-a4-megatron-queue-script:
	$(PYTHON) infra/runpod/runpod_queue.py --queue-config "$(RUNPOD_A4_MEGATRON_QUEUE_CONFIG)" $(RUNPOD_A4_MEGATRON_QUEUE_RUN_ARG) container-script

runpod-a4-megatron-pod-create-command:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required, for example RUN_ID=runpod-a4-megatron-$$(date -u +%Y%m%dT%H%M%SZ)"; exit 2)
	$(PYTHON) infra/runpod/runpod_queue.py --queue-config "$(RUNPOD_A4_MEGATRON_QUEUE_CONFIG)" $(RUNPOD_A4_MEGATRON_QUEUE_RUN_ARG) pod-create-command

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
