#!/usr/bin/env python3
"""Minimal Ulysses multi-GPU smoke: prove torch.distributed + init_parallel_state
(N /opt/bernini/src) form working process groups across 2 GPUs and a collective
completes. Run via torchrun --nproc-per-node=2 --standalone usp_smoke.py
(launched from inside the wan-worker container with the bernini venv)."""
import os
import torch
import torch.distributed as dist


def main() -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)

    from bernini.parallel import init_parallel_state, get_parallel_state
    init_parallel_state(ulysses_size=world)
    ps = get_parallel_state()

    # collective round-trip across ranks (proves NCCL group + veomni groups OK)
    t = torch.full((4,), float(rank), device=f"cuda:{local}", dtype=torch.float32)
    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t)
    dist.barrier()
    ok = all(int(x[0].item()) == i for i, x in enumerate(gathered))

    print(f"RANK{rank} USP_SMOKE world={world} ulysses={ps.ulysses_size} "
          f"enabled={ps.ulysses_enabled} dp={ps.dp_size} "
          f"ulp_rank={ps.ulysses_rank} allgather={ok} "
          f"mem_mb={torch.cuda.max_memory_allocated() // (1024 * 1024)}",
          flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
