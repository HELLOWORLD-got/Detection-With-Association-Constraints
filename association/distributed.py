"""torchrun setup, global batch sizing, and per-rank random-state checkpoints."""

from datetime import timedelta
import os
import random
import torch
import torch.distributed as dist


def setup(device_spec):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(device_spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable on this machine")
        if world_size > 1 and device.index is not None:
            raise ValueError("With torchrun use --device cuda; LOCAL_RANK selects each GPU")
        device = torch.device("cuda", local_rank if world_size > 1 else device.index or 0)
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Training supports CPU or CUDA")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2)
        )
    return device, rank, world_size


def per_rank_batch_size(global_batch_size, world_size):
    if global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError(
            f"Global --batch-size {global_batch_size} must be positive and divisible "
            f"by {world_size} processes. Use fewer GPUs or increase the global batch."
        )
    return global_batch_size // world_size


def capture_rng(device):
    return dict(
        torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def gather_rng(device, world_size):
    state = capture_rng(device)
    if world_size == 1:
        return [state]
    states = [None] * world_size
    dist.all_gather_object(states, state)
    return states


def restore_rng(checkpoint, device, rank, world_size):
    states = checkpoint.get("rank_rng_states")
    if states is not None:
        if len(states) != world_size:
            if rank == 0:
                print(
                    "Resume uses a different GPU/process count: weights and optimizer restored; "
                    "per-rank RNG streams restart, so this is not an exact replay.",
                    flush=True,
                )
            return
        state = states[rank]
    else:
        if world_size != 1:
            if rank == 0:
                print(
                    "Loading a legacy single-process checkpoint into DDP; RNG streams restart.",
                    flush=True,
                )
            return
        cuda_states = checkpoint.get("cuda_rng", [])
        state = dict(
            torch_rng=checkpoint["torch_rng"],
            python_rng=checkpoint["python_rng"],
            cuda_rng=cuda_states[0] if cuda_states else None,
        )
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if device.type == "cuda" and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
