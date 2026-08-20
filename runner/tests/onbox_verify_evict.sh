#!/usr/bin/env bash
# onbox_verify_evict.sh — verify that POST /evict clears the MODEL CUDA context
# from the GPU for each worker (image / ltx / idv2v) on the .151 box.
#
# MODEL-VRAM MODEL, not literal zero: a worker keeps its long-lived CUDA
# primary context (~0.5-1 GB, driver-reserved for the process lifetime) while
# it stays alive for warm lazy reload. /evict must return ALL MODEL tensors and
# the caching-allocator pool to the driver (drop refs -> torch.cuda.synchronize
# -> gc -> torch.cuda.empty_cache), leaving only that primary-context floor.
# So the success bar is FLOOR_MB (<=1 GB), NOT 0.
#
# Method: for each worker, find the container's host PID, record its idle
# floor, POST /load on GPU <NGPU>, POST /evict, and confirm the PID is back at
# (floor + slack) — model VRAM released, no leak.
#
# Runs the control-plane calls via docker exec (worker control ports are
# internal to the compose network, not host-published). Run on .151:
#   TOKEN=<worker-token> NGPU=<slot> bash /tmp/onbox_verify_evict.sh
set -uo pipefail

TOKEN="${TOKEN:?set TOKEN to the shared worker token}"
NGPU="${NGPU:-1}"
GRACE="${GRACE:-2}"            # seconds to settle after load/evict
FLOOR_MB="${FLOOR_MB:-1500}"   # primary-context floor + slack, in MiB
SVC_IMAGE="${SVC_IMAGE:-video-creator-image-worker}"
SVC_LTX="${SVC_LTX:-video-creator-ltx-worker}"
SVC_IDV2V="${SVC_IDV2V:-video-creator-idv2v-worker}"

now() { date +%H:%M:%S; }

pid_for() { docker inspect --format '{{.State.Pid}}' "$1"; }
port_for() { # service -> internal control port
  case "$1" in
    *image*) echo 8994;; *ltx*) echo 8991;; *idv2v*) echo 8992;; *) echo 8994;;
  esac
}

used_of() { # $1=PID -> total MiB the PID holds across all GPUs (0 if none)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    | awk -F', ' -v p="$1" '$1==p {s+=$2} END {print s+0}'
}

ctrl() { # $1=svc $2=endpoint $3=json-body -> docker exec curl to that worker
  docker exec "$1" curl -s -X POST "http://localhost:$(port_for "$1")/$2" \
    -H "X-Worker-Token: $TOKEN" -H 'Content-Type: application/json' -d "$3"
}

test_worker() { # $1=svc
  local svc="$1" pid floor before after
  pid=$(pid_for "$svc")
  floor=$(used_of "$pid")
  echo "[$(now)] $svc (pid=$pid): idle floor=${floor} MiB"
  ctrl "$svc" "load" "{\"device\": $NGPU}" >/dev/null 2>&1
  sleep "$GRACE"
  before=$(used_of "$pid")
  echo "[$(now)] $svc: after /load holds ${before} MiB"
  ctrl "$svc" "evict" "{\"device\": $NGPU}" >/dev/null 2>&1
  sleep "$GRACE"
  after=$(used_of "$pid")
  if [ "$after" -le "$FLOOR_MB" ]; then
    verdict="MODEL VRAM CLEARED (at floor ${after} MiB <= ${FLOOR_MB})"
  else
    verdict="STILL RESIDENT: ${after} MiB > floor ${FLOOR_MB}!"
  fi
  echo "[$(now)] $svc: after /evict holds ${after} MiB  ->  $verdict"
  # leave the worker resident again (re-load) so the live stack isn't degraded
  ctrl "$svc" "load" "{\"device\": $NGPU}" >/dev/null 2>&1 || true
}

echo "=== onbox verify /evict clears MODEL CUDA context (NGPU=$NGPU, floor<=${FLOOR_MB}MiB) $(now) ==="
test_worker "$SVC_IMAGE" || echo "[$(now)] image check issue"
test_worker "$SVC_LTX"   || echo "[$(now)] ltx check issue"
test_worker "$SVC_IDV2V" || echo "[$(now)] idv2v check issue"
echo "[$(now)] final GPU state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "[$(now)] DONE"
