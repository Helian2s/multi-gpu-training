# EXP-01: AWS PCIe P2P and NCCL communication — report

Report status: Not run

## Executive conclusion

Not run. This report remains a placeholder until AWS-A2 qualification and
EXP-01 collection complete.

## Run inventory

Link immutable run IDs and record excluded or failed runs with reasons.

## Environment

Record the EC2 instance ID, Availability Zone, instance type, GPU inventory,
`nvidia-smi topo -m`, NVIDIA driver, CUDA, NCCL, image digest, Git commit, and
qualification result.

## Method

Run the model-free collection script inside the PyTorch/NVIDIA-tools image on
one `g7e.12xlarge` with exactly two visible GPUs. Capture topology first, then
CUDA P2P bandwidth/latency, then NCCL collectives over the declared message-size
sweep.

## Results

To be filled after analysis:

- Peer-access matrix and P2P bandwidth/latency by direction.
- NCCL latency and bus bandwidth by collective and message size.
- NCCL algorithm/debug excerpts needed to explain the result.

## Interpretation

Explain the observed mechanism using profiler or communication evidence. Keep
observation separate from inference.

## Limitations and anomalies

Document variance, unavailable counters, fallbacks, failed cases, and threats to
comparability.

## Cost

Record complete billed instance duration, active GPU-hours, estimated
On-Demand cost, EBS/S3 costs if material, and termination time.

## Exam takeaway

State the configuration choice or diagnostic rule that should be remembered for
an NCP-GENL-style scenario.

## Reproduction

Use `experiments/exp_01_aws_pcie_p2p_nccl_communication/experiment.yaml` and
the AWS launch/qualification wrapper once the final image digest is recorded.
