#!/bin/bash
# Launch the Ulysses 2-rank smoke inside the wan-worker container on GPUs 1+2
# (CUDA_VISIBLE_DEVICES maps LOCAL_RANK 0/1 -> physical GPU 1/2, so the
# production bernini resident on GPU 0 is untouched).
set -e
docker cp /tmp/usp_smoke.py video-creator-wan-worker:/tmp/usp_smoke.py
echo "=== torchrun 2-rank USP smoke (GPUs 1,2) ==="
docker exec -e CUDA_VISIBLE_DEVICES=1,2 -e PYTHONPATH=/opt/bernini/src \
  video-creator-wan-worker \
  /opt/bernini/venv/bin/python -m torch.distributed.run \
    --nproc-per-node=2 --standalone --master-port=29555 \
    /tmp/usp_smoke.py
echo "EXIT:$?"
