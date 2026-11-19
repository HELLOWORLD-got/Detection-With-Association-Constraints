import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image
from association.data import MultiViewDataset
from association.metrics import AssociationMetrics, DetectionMetrics
from association.models import InstanceBasedAssociation
from association.losses import HungarianLoss
from tools.prepare_data import prepare


def target():
    return dict(
        boxes=torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]] * 2),
        visible=torch.ones(2, 2, dtype=torch.bool),
        labels=torch.zeros(2, dtype=torch.long),
        sequence="clip",
        sizes=[[100, 100], [100, 100]],
    )


def prediction(t):
    return dict(
        boxes=t["boxes"].clone(),
        present=t["visible"].clone(),
        labels=t["labels"].clone(),
        scores=torch.ones(2, 2),
    )


class MetricsTests(unittest.TestCase):
    def test_perfect_and_slot_permutation(self):
        t = target()
        p = prediction(t)
        for key in ("boxes", "present", "scores"):
            p[key] = p[key][:, [1, 0]]
        p["labels"] = p["labels"][[1, 0]]
        m = AssociationMetrics()
        m.update(p, t)
        r = m.compute()
        self.assertEqual(r["MDA"], 1.0)
        self.assertEqual(r["UCR"], 1.0)
        d = DetectionMetrics(1)
        d.update(p, t)
        self.assertAlmostEqual(d.compute()["mAP"], 1.0)

    def test_swapped_association_fails_despite_perfect_detections(self):
        t = target()
        p = prediction(t)
        p["boxes"][1] = p["boxes"][1, [1, 0]]
        m = AssociationMetrics()
        m.update(p, t)
        self.assertEqual(m.compute()["MDA"], 0.0)
        self.assertEqual(m.compute()["UCR"], 0.0)

    def test_visibility_false_positive_breaks_ucr(self):
        t = target()
        t["visible"][1, 0] = False
        p = prediction(t)
        p["present"][1, 0] = True
        m = AssociationMetrics()
        m.update(p, t)
        self.assertEqual(m.compute()["UCR"], 0.5)

    def test_duplicate_slots_cannot_inflate_true_associations(self):
        t = target()
        p = prediction(t)
        p["boxes"] = torch.cat([p["boxes"], p["boxes"][:, :1]], 1)
        p["present"] = torch.ones(2, 3, dtype=torch.bool)
        p["labels"] = torch.zeros(3, dtype=torch.long)
        p["scores"] = torch.ones(2, 3)
        m = AssociationMetrics()
        m.update(p, t)
        r = m.compute()
        self.assertEqual(r["true_associations"], 2)
        self.assertAlmostEqual(r["MDA"], 2 / 3)

    def test_absent_zero_area_boxes_do_not_create_nan(self):
        t = target()
        t["visible"][1, 0] = False
        t["boxes"][1, 0] = 0
        p = prediction(t)
        m = AssociationMetrics()
        m.update(p, t)
        self.assertEqual(m.compute()["UCR"], 1.0)

    def test_empty_predictions(self):
        t = target()
        p = prediction(t)
        p["present"][:] = False
        p["scores"][:] = 0
        m = AssociationMetrics()
        m.update(p, t)
        self.assertEqual(m.compute()["UCR"], 0.0)
        d = DetectionMetrics(1)
        d.update(p, t)
        self.assertEqual(d.compute()["mAP"], 0.0)


class DataTests(unittest.TestCase):
    def test_filename_offset_alignment_and_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "raw"
            out = Path(temp) / "prepared"
            for split, clip in [("train", "1"), ("val", "2"), ("test", "3")]:
                for v in (1, 2):
                    d = root / split / str(v) / f"{clip}-{v}"
                    d.mkdir(parents=True)
                    Image.new("RGB", (100, 80)).save(d / "00000201.jpg")
                    x = root / "new_xml" / str(v)
                    x.mkdir(parents=True, exist_ok=True)
                    (x / f"{clip}-{v}.xml").write_text(
                        '<annotations><track id="7" label="car"><box frame="0" outside="0" occluded="1" xtl="10" ytl="20" xbr="30" ybr="40"/></track></annotations>'
                    )
            with contextlib.redirect_stdout(io.StringIO()):
                r = prepare(root, out)
            ds = MultiViewDataset(out / "train.jsonl", num_queries=1, num_classes=1)
            images, t = ds[0]
            self.assertTrue(t["visible"].all())
            self.assertEqual(t["ids"].tolist(), [7])
            self.assertEqual(ds.record(0)["image_number"], 201)
            self.assertTrue(torch.allclose(t["boxes"][0, 0], torch.tensor([0.2, 0.375, 0.2, 0.25])))
            with self.assertRaisesRegex(ValueError, "No annotations were truncated"):
                MultiViewDataset(out / "train.jsonl", num_queries=0)
            self.assertFalse(r["sequence_leakage"])

    def test_no_cross_split_clip_leakage(self):
        # Pair/clip leakage has a separate audit guard; no frame-level random split is used.
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "x.jsonl"
            p.write_text(
                json.dumps(dict(images=["a", "b"], labels=[0] * 26, boxes=[], visible=[])) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "26 global instances"):
                MultiViewDataset(p, num_queries=25)

    def test_logits_loss_remains_finite_for_extreme_scores(self):
        out = dict(
            pred_logits=torch.tensor([[[1000.0, -1000.0], [-1000.0, 1000.0]]], requires_grad=True),
            pred_boxes=torch.full((1, 2, 2, 4), 0.2, requires_grad=True),
            pred_visibility_logits=torch.tensor([[[1000.0, -1000.0]] * 2], requires_grad=True),
        )
        t = dict(
            labels=torch.tensor([0]),
            boxes=torch.full((2, 1, 4), 0.2),
            visible=torch.ones(2, 1, dtype=torch.bool),
        )
        loss = HungarianLoss()(out, [t])["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(out["pred_logits"].grad).all())

    def test_nonfinite_matching_reports_sample_and_tensor(self):
        boxes = torch.full((1, 2, 2, 4), 0.2)
        boxes[0, 0, 0, 0] = float("nan")
        outputs = dict(
            pred_logits=torch.zeros(1, 2, 2),
            pred_boxes=boxes,
            pred_visibility_logits=torch.zeros(1, 2, 2),
        )
        target = dict(
            labels=torch.tensor([0]), boxes=torch.full((2, 1, 4), 0.2),
            visible=torch.ones(2, 1, dtype=torch.bool), sequence="23", frame=334,
        )
        with self.assertRaisesRegex(FloatingPointError, "sequence=23, frame=334.*pred_boxes"):
            HungarianLoss()(outputs, [target])


if __name__ == "__main__":
    unittest.main()
