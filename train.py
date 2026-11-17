"""Single GPU/CPU or torchrun DDP training, validation, and epoch-boundary resume.
--batch-size is the GLOBAL micro-batch across all ranks (also for single GPU).
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from association import distributed
from association.engine import (
    make_loader,
    load_config,
    seed_all,
    build_model,
    freeze_bn_stats,
    criterion_from,
    outputs_float,
    evaluate_model,
)


def save_checkpoint(path, model, optimizer, scaler, epoch, best, config, rng_states, world_size):
    # Save the underlying model, so single-GPU evaluate/infer need no DDP prefix stripping.
    value = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        epoch=epoch,
        best=best,
        config=config,
        world_size=world_size,
        rank_rng_states=rng_states,
        torch_rng=rng_states[0]["torch_rng"],
        python_rng=rng_states[0]["python_rng"],
        cuda_rng=[rng_states[0]["cuda_rng"]] if rng_states[0]["cuda_rng"] is not None else [],
    )
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def run(args, config, device, rank, world_size):
    t = config["training"]
    local_batch = distributed.per_rank_batch_size(t["batch_size"], world_size)
    t["amp"] = t["amp"] and device.type == "cuda" and not args.no_amp
    seed_all(t["seed"] + rank, t["deterministic"])
    train = make_loader(
        config,
        "train",
        args.limit_train,
        True,
        config["data"]["augment"],
        world_size=world_size,
        rank=rank,
        batch_size=local_batch,
    )
    # Validation stays unsharded on rank 0: no repeated/padded GT and no AP averaging.
    val = make_loader(config, "val", args.limit_val) if rank == 0 else None
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume and resume["config"]["model"] != config["model"]:
        raise ValueError("Resume model configuration differs from checkpoint")
    raw_model = build_model(config, pretrained=False if resume else None).to(device)
    if resume:
        raw_model.load_state_dict(resume["model"])
    model = (
        DistributedDataParallel(
            raw_model, device_ids=[device.index] if device.type == "cuda" else None
        )
        if world_size > 1
        else raw_model
    )
    criterion = criterion_from(config)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"])
    start, best = 0, -float("inf")
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        elif t["amp"] and rank == 0:
            print("Checkpoint has no AMP state; initializing a fresh GradScaler.", flush=True)
        start, best = resume["epoch"], resume["best"]
        distributed.restore_rng(resume, device, rank, world_size)
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2))
        (out / "invocation.json").write_text(json.dumps(vars(args), indent=2))
        topology = dict(
            world_size=world_size,
            backend=dist.get_backend() if world_size > 1 else None,
            global_micro_batch=t["batch_size"],
            per_rank_micro_batch=local_batch,
            accumulation_steps=t["accumulation_steps"],
            effective_batch=t["batch_size"] * t["accumulation_steps"],
            workers_per_rank=t["workers"],
            training_samples=len(train.dataset),
            padded_training_samples_per_epoch=(
                (len(train.sampler) * world_size - len(train.dataset)) if world_size > 1 else 0
            ),
            validation="full validation set on rank 0 only",
        )
        (out / "distributed.json").write_text(json.dumps(topology, indent=2))
        print(json.dumps(topology), flush=True)
    for epoch in range(start, t["epochs"]):
        if isinstance(train.sampler, DistributedSampler):
            train.sampler.set_epoch(epoch)
        model.train()
        if t["freeze_bn_stats"]:
            freeze_bn_stats(raw_model)
        optimizer.zero_grad(set_to_none=True)
        total, samples = 0.0, 0
        began = time.monotonic()
        batches = min(len(train), args.max_train_batches) if args.max_train_batches else len(train)
        for index, (images, targets) in enumerate(train):
            if index >= batches:
                break
            group_start = (index // t["accumulation_steps"]) * t["accumulation_steps"]
            group_size = min(t["accumulation_steps"], batches - group_start)
            step_now = (index + 1) % t["accumulation_steps"] == 0 or index + 1 == batches
            # no_sync must surround BOTH forward and backward.
            sync_context = model.no_sync() if world_size > 1 and not step_now else nullcontext()
            with sync_context:
                with torch.autocast(device_type=device.type, enabled=t["amp"]):
                    outputs = model([x.to(device) for x in images], return_logits=True)
                loss = criterion(outputs_float(outputs), targets, sync_distributed=world_size > 1)[
                    "loss"
                ]
                finite = torch.isfinite(loss).to(dtype=torch.int32)
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("Non-finite loss on at least one rank")
                scaler.scale(loss / group_size).backward()
            if step_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), t["clip_grad_norm"], error_if_nonfinite=not t["amp"]
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item() * len(targets)
            samples += len(targets)
            if rank == 0 and index % 100 == 0:
                print(
                    f"epoch={epoch+1} batch={index+1}/{batches} loss={loss.item():.5f}", flush=True
                )
        totals = torch.tensor([total, samples], device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(totals)
        result, improved = None, False
        if rank == 0:
            # Never call the DDP wrapper here: other ranks wait, not validate.
            result = evaluate_model(raw_model, val, criterion, config, device)
            selected = result[config["evaluation"]["selection_metric"]]
            improved = selected > best
            if improved:
                best = selected
        if world_size > 1:
            best_tensor = torch.tensor(best, device=device, dtype=torch.float64)
            dist.broadcast(best_tensor, src=0)
            best = best_tensor.item()
        # Gather after validation so each rank's next-epoch RNG can be restored.
        rng_states = distributed.gather_rng(device, world_size)
        if rank == 0:
            record = dict(
                epoch=epoch + 1,
                train_loss=(totals[0] / totals[1].clamp_min(1)).item(),
                train_samples=int(totals[1].item()),
                validation=result,
                elapsed_seconds=time.monotonic() - began,
                best=best,
            )
            with (out / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            save_checkpoint(
                out / "last.pt",
                raw_model,
                optimizer,
                scaler,
                epoch + 1,
                best,
                config,
                rng_states,
                world_size,
            )
            if improved:
                save_checkpoint(
                    out / "best.pt",
                    raw_model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    best,
                    config,
                    rng_states,
                    world_size,
                )
            print(json.dumps(record), flush=True)
        if world_size > 1:
            dist.barrier()
    if rank == 0 and start >= t["epochs"]:
        print(f"Checkpoint already completed {start} epochs; increase --epochs to continue.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--config", default="configs/resnet50.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/mdmt")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="GLOBAL micro-batch across all GPUs")
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--workers", type=int, help="DataLoader workers PER rank")
    parser.add_argument(
        "--limit-train", type=int, help="Total dataset limit BEFORE distributed sharding"
    )
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--max-train-batches", type=int, help="Maximum batches PER rank")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("epochs", "batch_size", "accumulation_steps", "workers"):
        if getattr(args, name) is not None:
            config["training"][name] = getattr(args, name)
    for name in ("epochs", "batch_size", "accumulation_steps"):
        if config["training"][name] < 1:
            parser.error(f"{name} must be positive")
    for name in ("limit_train", "limit_val", "max_train_batches"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'--{name.replace("_","-")} must be positive')
    try:
        device, rank, world_size = distributed.setup(args.device)
        run(args, config, device, rank, world_size)
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
