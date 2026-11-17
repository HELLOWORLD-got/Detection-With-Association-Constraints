import json
from pathlib import Path
import tempfile
import unittest
from association.distributed import per_rank_batch_size
from association.engine import make_loader


class DistributedInputTests(unittest.TestCase):
    def test_global_batch_preserved_and_invalid_topology_rejected(self):
        self.assertEqual(per_rank_batch_size(4, 2), 2)
        self.assertEqual(per_rank_batch_size(4, 4), 1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            per_rank_batch_size(4, 8)

    def test_rank_shards_disjoint_and_epoch_shuffle_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                "".join(json.dumps(dict(images=["a", "b"], labels=[0])) + "\n" for _ in range(8))
            )
            config = dict(
                data=dict(train=str(path), height=64, width=128),
                model=dict(num_views=2, num_queries=4, num_classes=1),
                training=dict(batch_size=4, workers=0, seed=42),
            )
            loaders = [
                make_loader(config, "train", shuffle=True, world_size=2, rank=r, batch_size=2)
                for r in (0, 1)
            ]
            shards = [list(loader.sampler) for loader in loaders]
            self.assertFalse(set(shards[0]) & set(shards[1]))
            self.assertEqual(set(shards[0]) | set(shards[1]), set(range(8)))
            for loader in loaders:
                loader.sampler.set_epoch(1)
            self.assertNotEqual(shards, [list(loader.sampler) for loader in loaders])
