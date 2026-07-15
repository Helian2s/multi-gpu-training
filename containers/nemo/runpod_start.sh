#!/usr/bin/env bash
set -euo pipefail

echo "Runpod start script active."

mkdir -p /workspace /runpod-volume/artifacts/runs

printenv | grep -E '^[A-Z_][A-Z0-9_]*=' | grep -v '^PUBLIC_KEY=' | \
  awk -F= '{ val = $0; sub(/^[^=]*=/, "", val); gsub(/"/, "\\\"", val); print "export " $1 "=\"" val "\"" }' \
  > /etc/rp_environment || true
if ! grep -q 'source /etc/rp_environment' /root/.bashrc 2>/dev/null; then
  echo 'source /etc/rp_environment' >> /root/.bashrc
fi

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  echo "Configuring SSH access."
  mkdir -p /root/.ssh /run/sshd
  printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A
  /usr/sbin/sshd -e
else
  echo "PUBLIC_KEY is not set; SSH will not be started."
fi

echo "Pod is ready; keeping container alive."
exec sleep infinity
