#!/usr/bin/env bash
# tunnel.sh — keep SSH port-forwards alive from OVH → NeuroSpin cluster
#
# Forwards:
#   localhost:8000 → cluster:8000  (vLLM)
#   localhost:6333 → cluster:6333  (Qdrant)
#
# Requires autossh:  apt install autossh
# Edit CLUSTER_USER and CLUSTER_HOST, or export them as env vars.
#
# Usage:
#   ./tunnel.sh             runs in foreground (Ctrl-C to stop)
#   ./tunnel.sh &           runs in background
#   See tunnel.service for a proper systemd unit.

set -euo pipefail

CLUSTER_USER="${CLUSTER_USER:-cbel}"
CLUSTER_HOST="${CLUSTER_HOST:-ext1.idris.fr}"

echo "Starting SSH tunnel to ${CLUSTER_USER}@${CLUSTER_HOST}…"
echo "  localhost:8000 → vLLM on cluster"
echo "  localhost:6333 → Qdrant on cluster"
echo "Press Ctrl-C to stop."

exec autossh \
  -M 0 \
  -N \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=accept-new" \
  -L "8000:localhost:8000" \
  -L "6333:localhost:6333" \
  "${CLUSTER_USER}@${CLUSTER_HOST}"
