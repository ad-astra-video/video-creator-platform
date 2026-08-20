#!/usr/bin/env bash
# onbox_verify_evict.sh — verify that POST /evict clears the CUDA context from
# the GPU for each worker (image / ltx / idv2v) on the .151 box.
#
# Method: for each worker service, find the container's host PID, have it load
# a model on GPU <DEV>, confirm the PID is a compute-app holding memory on that
# GPU (nvidia-smi --query-compute-apps), then POST /evict and confirm the PID
# no longer holds ANY compute-app memory on that GPU (context cleared) even
# though the process stays alive for hot reload.
#
# Usage (on .151, from the compose dir):
#   TOKEN=<worker-token> NGPU=<slot> bash runner/tests/onbox_verify_evict.sh
# NGPU defaults to 0. Set per-service token if not shared.
set -euo pipefail

TOKEN="${TOKEN:?set TOKEN to the shared worker token}"
NGPU="${NGPU:-0}"
COMPOSE="docker-compose.video-creator.yml"
GRACE="${GRACE:-3}"          # seconds to let the model load
: ${SVC_IMAGE:=video-creator-image-worker}
: ${SVC_LTX:=video-creator-ltx-worker}
: ${SVC_IDV2V:=video-creator-idv2v-worker}

now() { date +%H:%M:%S; }

pid_for() { # $1=service -> host PID of the container's main process
  docker inspect --format '{{.State.Pid}}' "$1"
}

used_on_gpu() { # $1=PID $2=GPU index -> MB the PID holds on that GPU (0 if none)
  local pid="$1" gpu="$2" uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
           | awk -F', ' -v g="$gpu" '$1==g {print $2}')
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    | awk -F', ' -v p="$pid" '$1==p {sum+=$2} END {print sum+0}'
}

evict_and_check() { # $1=service $2=port $3=GPU
  local svc="$1" port="$2" gpu="$3" pid bef aft
  pid=$(pid_for "$svc")
  echo "[$(now)] $svc: host PID=$pid — /load on GPU $gpu ..."
  curl -sf -X POST "http://localhost:$port/load" \
    -H "X-Worker-Token: $TOKEN" -H 'Content-Type: application/json' \
    -d "{\"device\": $gpu}" >/dev/null
  sleep "$GRACE"
  bef=$(used_on_gpu "$pid" "$gpu")
  echo "[$(now)] $svc: after /load, PID holds ${bef} MB on GPU $gpu"
  if [ "$bef" -le "${MIN_LOADED_MB:-200}" ]; then
    echo "  !! WARN: PID holds ~0 MB after load — model may not be resident" >&2
  fi
  curl -sf -X POST "http://localhost:$port/evict" \
    -H "X-Worker-Token: $TOKEN" -H 'Content-Type: application/json' \
    -d "{\"device\": $gpu}" >/dev/null
  sleep "$GRACE"
  aft=$(used_on_gpu "$pid" "$gpu")
  echo "[$(now)] $svc: after /evict, PID holds ${aft} MB on GPU $gpu  → $([ "$aft" -le 0 ] && echo 'CONTEXT CLEARED' || echo 'STILL RESIDENT!')"
}

echo "=== onbox verify /evict clears CUDA context (NGPU=$NGPU) $(now) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

evict_and_check "$SVC_IMAGE" 8994 "$NGPU"   || echo "[$(now)] image worker check FAILED"
evict_and_check "$SVC_LTX"   8991 "$NGPU"   || echo "[$(now)] ltx worker check FAILED"
evict_and_check "$SVC_IDV2V" 8992 "$NGPU"   || echo "[$(now)] idv2v worker check FAILED"

echo "[$(now)] final GPU state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "[$(now)] DONE"
