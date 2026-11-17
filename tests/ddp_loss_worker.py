"""Run with torchrun --nproc-per-node=2; compare DDP gradients to a global batch.
One rank has visible GT, the other has none. No dropout or optimizer approximation.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from association import distributed
from association.losses import HungarianLoss


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[[1.0, -0.2], [-0.3, 0.8]]]))
        self.boxes = nn.Parameter(torch.full((1, 2, 2, 4), 0.3))
        self.visibility = nn.Parameter(torch.zeros(1, 2, 2))

    def forward(self, batch=1):
        return dict(
            pred_logits=self.logits.expand(batch, -1, -1),
            pred_boxes=self.boxes.expand(batch, -1, -1, -1),
            pred_visibility_logits=self.visibility.expand(batch, -1, -1),
        )


if __name__ == "__main__":
    device, rank, world = distributed.setup("cpu")
    try:
        assert world == 2
        model = Heads()
        wrapped = DistributedDataParallel(model)
        targets = [
            dict(
                labels=torch.tensor([0]),
                boxes=torch.full((2, 1, 4), 0.2),
                visible=torch.ones(2, 1, dtype=torch.bool),
            ),
            dict(
                labels=torch.empty(0, dtype=torch.long),
                boxes=torch.empty(2, 0, 4),
                visible=torch.empty(2, 0, dtype=torch.bool),
            ),
        ]
        criterion = HungarianLoss()
        loss = criterion(wrapped(), [targets[rank]], sync_distributed=True)["loss"]
        loss.backward()
        reference = Heads()
        expected = criterion(reference(2), targets)["loss"]
        expected.backward()
        errors = {
            name: float((param.grad - dict(reference.named_parameters())[name].grad).abs().max())
            for name, param in model.named_parameters()
        }
        assert all(error < 1e-6 for error in errors.values()), errors
        if rank == 0:
            result = dict(
                global_batch_gradient_match=True,
                maximum_gradient_errors=errors,
                test_case="visible targets on rank0, empty target set on rank1",
            )
            Path("runs/ddp_loss_check.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        distributed.cleanup()
