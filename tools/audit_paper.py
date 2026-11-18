"""Reproduce concrete gaps between paper equations and unmodified author code.
No training, data modification, or replacement attention operator is performed.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn


def attempt(fn):
    try:
        return dict(completed=True, result=fn())
    except Exception as error:
        return dict(completed=False, error_type=type(error).__name__, error=str(error))


def audit():
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1] / "reference/author_model"
    giq = runpy.run_path(str(root / "global_instance_query.py"))["GlobalInstanceQuery"]
    mvff = runpy.run_path(str(root / "multi_view_feature_fusion.py"))["MultiViewFeatureFusion"]
    mvoq = runpy.run_path(str(root / "multi_view_object_query.py"))["MultiViewObjectQuery"]
    features = [torch.randn(1, 32, 2, 2) for _ in range(2)]

    def original_giq():
        giq(hidden_dim=32, num_queries=2)
        return "constructed"

    def original_mvff():
        return list(mvff(hidden_dim=32, num_layers=1)(features).shape)

    def original_reshape():
        model = mvff(hidden_dim=32, num_layers=1)
        model.ca_layers = nn.ModuleList()
        model.sa_layers = nn.ModuleList()
        return list(model(features).shape)

    def triplet_shapes():
        # Eq.11 identifies one instance vector as K and H'W' image tokens as V.
        return list(nn.MultiheadAttention(32, 8, batch_first=True)(
            torch.randn(1, 1, 32), torch.randn(1, 1, 32), torch.randn(1, 4, 32)
        )[0].shape)

    @torch.no_grad()
    def original_slot_order():
        model = mvoq(hidden_dim=32, num_queries=2).eval()
        boxes, _ = model(torch.randn(1, 2, 32), [features[0]])
        difference = (boxes[0][:, 0] - boxes[0][:, 1]).abs().max().item()
        return dict(max_slot_box_difference=difference, distinct_input_instances=True,
                    setup="N=2, H'W'=4; source repeat/reshape mixes instance order",
                    note="Counterexample in public code, not proof of the server model's root cause.")

    return dict(
        source=json.loads((root / "SOURCES.json").read_text()),
        checks={
            "author_giq_position_encoding": attempt(original_giq),
            "author_mvff_encoder_called_as_qkv": attempt(original_mvff),
            "author_mvff_reshape_after_skipping_attention": attempt(original_reshape),
            "paper_eq11_standard_attention_shape_check": attempt(triplet_shapes),
            "author_mvoq_instance_order_counterexample": attempt(original_slot_order),
        },
        unresolved=[
            "Paper PDF p6: channel concat produces 256*K channels; SA hidden/output are 256, reduction is unspecified.",
            "Paper PDF p7 Eq11: object query Q, single instance K, spatial view V; token alignment/projection is unspecified.",
            "Paper N=25 versus complete MDMT union-instance maxima train191/val221/test201; selection/cropping protocol is unspecified.",
            "Loss weights and several implementation/training choices are not supplied.",
        ],
        conclusion="Exact reproduction is not established. Do not label an invented attention/reduction as the paper architecture.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/reports/paper_architecture_audit.json")
    args = parser.parse_args()
    report = audit()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
