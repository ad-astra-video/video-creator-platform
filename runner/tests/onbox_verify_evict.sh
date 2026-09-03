#!/usr/bin/env bash
# onbox_verify_evict.sh — verify that POST /evict clears the FULL worker off the
# GPUs for each worker (image / ltx / idv2v) on the .151 box.
#
# New (2026) contract after subprocess isolation: the GPU model lives in a child
# subprocess spawned by the worker's aiohttp server. /evict TERMINATES that child
# (process exit destroys its CUDA primary context — PyTorch has no in-process API
# to do this), leaving the parent server CUDA-free. Success bar is now ZERO: after
# /evict, NO process belonging to the worker container may hold ANY GPU memory.
#
# Method: for each worker, capture the container's full process set
# (`docker top` = main + child PIDs). Record its GPU memory across the load/evict
# cycle. After /evict, the container's GPU-held process set must be EMPTY.
#
# Runs the control-plane calls via docker exec (worker control ports are internal
# to the compose network, not host-published). Run on .151:
#   TOKEN=<worker-token> NGPU=<slot> bash /tmp/onbox_verify_evict.sh
set -uo pipefail

TOKEN="${TOKEN:?set TOKEN to the shared worker token}"
NGPU="${NGPU:-1}"
GRACE="${GRACE:-4}"            # seconds to settle after load/evict
SVC_IMAGE="${SVC_IMAGE:-video-creator-image-worker}"
SVC_LTX="${SVC_LTX:-video-creator-ltx-worker}"
SVC_IDV2V="${SVC_IDV2V:-video-creator-idv2v-worker}"

now() { date +%H:%M:%S; }

port_for() { # service -> internal control port
  case "$1" in
    *image*) echo 8994;; *ltx*) echo 8991;; *idv2v*) echo 8992;; *) echo 8994;;
  esac
}

# All host PIDs belonging to a container (main + spawned children).
pids_of() {
  docker top "$1" -o pid 2>/dev/null | tail -n +2 | sed 's/^ *//'
}

# PIDs that currently hold GPU memory, mapped to the container's process set.
# Returns lines "pid used_mb" for only those PIDs that both (a) have GPU memory
# and (b) belong to the container.
gpu_used_of() { # $1=svc -> "pid used_mb ..." for container-owned GPU holders
  local svc="$1" pids
  pids=$(pids_of "$svc")
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    | awk -F', ' -v set="$pids" '
        function in_set(p, s){ n=split(s,a,"\n"); for(i=1;i<=n;i++) if(a[i]==p) return 1; return 0 }
        { mb=$2; sub(/ MiB/,"",mb); if(in_set($1, set)) printf "%s %s\n", $1, mb }'
}

# Total MiB the container's own PIDs hold across all GPUs.
used_of() { gpu_used_of "$1" | awk '{s+=$2} END {print s+0}'; }

ctrl() { # $1=svc $2=endpoint $3=json-body -> docker exec curl to that worker
  docker exec "$1" curl -s -X POST "http://localhost:$(port_for "$1")/$2" \
    -H "X-Worker-Token: $TOKEN" -H 'Content-Type: application/json' -d "$3"
}

test_worker() { # $1=svc
  local svc="$1" after before holders
  if ! docker ps --format '{{.Names}}' | grep -q "^$svc$"; then
    echo "[$(now)] $svc: NOT RUNNING — skipped"; return
  fi
  # If the idv2v worker container is named differently on the box, allow an
  # env override; bail gracefully if it genuinely doesn't exist.
  echo "[$(now)] $svc: pre-load GPU holders: $(used_of "$svc") MiB (containers, not the child yet)"
  ctrl "$svc" "load" "{\"device\": $NGPU}" >/dev/null 2>&1
  sleep "$GRACE"
  before=$(used_of "$svc")
  echo "[$(now)] $svc: after /load holds ${before} MiB"
  ctrl "$svc" "evict" "{\"device\": $NGPU}" >/dev/null 2>&1
  sleep "$GRACE"
  after=$(used_of "$svc")
  holders=$(gpu_used_of "$svc")
  if [ -z "$holders" ]; then
    verdict="FULLY CLEARED — worker holds 0 MiB on every GPU (0 holder PIDs)"
  else
    verdict="STILL RESIDENT: ${after} MiB held by: $(echo "$holders" | tr '\n' ';')"
  fi
  echo "[$(now)] $svc: after /evict -> $verdict"
  # leave the worker resident again (re-load) so the live stack isn't degraded
  ctrl "$svc" "load" "{\"device\": $NGPU}" >/dev/null 2>&1 || true
}

echo "=== onbox verify /evict clears FULL worker off GPUs (NGPU=$NGPU) $(now) ==="
test_worker "$SVC_IMAGE"
test_worker "$SVC_LTX"
test_worker "$SVC_IDV2V" || echo "[$(now)] idv2v check issue"
echo "[$(now)] final GPU state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "[$(now)] DONE"
