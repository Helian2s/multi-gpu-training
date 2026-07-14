#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-/workspace/artifacts/runs/EXP-01/${RUN_ID}}"
RAW_DIR="${RUN_DIR}/raw"
METRICS_DIR="${RUN_DIR}/metrics"

mkdir -p "${RAW_DIR}" "${METRICS_DIR}"

exec > >(tee "${RAW_DIR}/collect_exp01.log") 2>&1

echo "EXP-01 collection started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"

env | sort > "${RAW_DIR}/environment.txt"

for required in nvidia-smi p2pBandwidthLatencyTest all_reduce_perf \
  reduce_scatter_perf all_gather_perf broadcast_perf alltoall_perf; do
  if ! command -v "${required}" >/dev/null 2>&1; then
    echo "missing required command: ${required}" >&2
    exit 2
  fi
done

nvidia-smi -L | tee "${RAW_DIR}/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "${RAW_DIR}/nvidia-smi-topo-m.txt"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total \
  --format=csv | tee "${RAW_DIR}/nvidia-smi-gpu-query.csv"

python3 - <<'PY' | tee "${RAW_DIR}/python-runtime.txt"
import json
import os
import torch

payload = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [
        {
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "capability": torch.cuda.get_device_capability(idx),
        }
        for idx in range(torch.cuda.device_count())
    ],
}
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["device_count"] != 2:
    raise SystemExit(f"expected exactly 2 CUDA devices, got {payload['device_count']}")
PY

echo "Running p2pBandwidthLatencyTest"
p2pBandwidthLatencyTest | tee "${RAW_DIR}/p2pBandwidthLatencyTest.txt"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL,GRAPH}"

MIN_BYTES="${NCCL_MIN_BYTES:-8}"
MAX_BYTES="${NCCL_MAX_BYTES:-1073741824}"
STEP_FACTOR="${NCCL_STEP_FACTOR:-2}"
WARMUP_ITERS="${NCCL_WARMUP_ITERS:-20}"
ITERS="${NCCL_ITERS:-100}"
CHECK_ITERS="${NCCL_CHECK_ITERS:-1}"
TIMEOUT_SECONDS="${NCCL_TIMEOUT_SECONDS:-900}"

run_nccl_test() {
  local name="$1"
  local binary="$2"
  local output="${RAW_DIR}/${name}.txt"

  echo "Running ${binary}"
  "${binary}" \
    --ngpus 2 \
    --minbytes "${MIN_BYTES}" \
    --maxbytes "${MAX_BYTES}" \
    --stepfactor "${STEP_FACTOR}" \
    --warmup_iters "${WARMUP_ITERS}" \
    --iters "${ITERS}" \
    --check "${CHECK_ITERS}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --average 1 \
    --report_cputime 1 \
    2>&1 | tee "${output}"
}

run_nccl_test nccl_all_reduce all_reduce_perf
run_nccl_test nccl_reduce_scatter reduce_scatter_perf
run_nccl_test nccl_all_gather all_gather_perf
run_nccl_test nccl_broadcast broadcast_perf
run_nccl_test nccl_all_to_all alltoall_perf

cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "experiment_id": "EXP-01",
  "run_id": "${RUN_ID}",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "raw_dir": "${RAW_DIR}",
  "metrics_dir": "${METRICS_DIR}"
}
EOF

echo "EXP-01 collection finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
