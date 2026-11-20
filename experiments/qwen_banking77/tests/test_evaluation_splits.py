"""A train/validation pilot must not read or evaluate held-out test examples."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fl_exp.trainer import ClientData, evaluate


PARTITION = "dirichlet_alpha_0.1"


class EvaluationSplitsTest(unittest.TestCase):
    def test_training_seeds_share_fixed_data_and_eval_samples_but_change_minibatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "seed_2026"
            prepared.mkdir()
            (prepared / "manifest.json").write_text(json.dumps({"seed": 2026, "n_clients": 10, "n_labels": 77}))
            for client in range(10):
                directory = prepared / PARTITION / "clients" / ("client_{:02d}".format(client))
                directory.mkdir(parents=True)
                for split in ("train", "validation"):
                    rows = [{"id": f"{split}:{client}:{index}", "text": "example", "label": index % 77}
                            for index in range(64)]
                    (directory / (split + ".jsonl")).write_text("\n".join(json.dumps(row) for row in rows))

            def tokenize(texts, **kwargs):
                return {key: torch.ones(len(texts), 2, dtype=torch.long)
                        for key in ("input_ids", "attention_mask")}

            datasets, batches = [], []
            for seed in (1, 2, 3, 4, 5, 1):
                config = {"seed": seed, "partition": PARTITION,
                          "model": {"num_labels": 77, "max_length": 2},
                          "data": {"seed": 2026, "prepared_root": str(root), "raw_dir": str(root / "raw")},
                          "training": {"eval_max_samples_per_client": 8,
                                       "evaluation_splits": ["train", "validation"]}}
                with patch("fl_exp.trainer.verify_prepared"):
                    data = ClientData(config, tokenize)
                datasets.append(data)
                batches.append(data.sample(0, 32))
                self.assertEqual(data.data_seed, 2026)
                self.assertEqual(data.evaluation_manifest(), datasets[0].evaluation_manifest())
                for client in range(10):
                    self.assertEqual(data.clients[client]["train"]["ids"], datasets[0].clients[client]["train"]["ids"])
            self.assertEqual(batches[0], batches[-1])
            self.assertEqual(len({tuple(batch) for batch in batches[:5]}), 5)

    def test_pilot_does_not_require_test_files_and_evaluates_full_client_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "seed_2026"
            prepared.mkdir()
            (prepared / "manifest.json").write_text(json.dumps({"seed": 2026, "n_clients": 10, "n_labels": 77}))
            for client in range(10):
                directory = prepared / PARTITION / "clients" / ("client_{:02d}".format(client))
                directory.mkdir(parents=True)
                for split in ("train", "validation"):
                    records = [{"id": "{}:{}:{}".format(split, client, index), "text": "example", "label": 0}
                               for index in range(client + 1)]
                    (directory / (split + ".jsonl")).write_text("\n".join(json.dumps(row) for row in records))

            def tokenize(texts, **kwargs):
                return {key: torch.ones(len(texts), 2, dtype=torch.long)
                        for key in ("input_ids", "attention_mask")}

            config = {"seed": 2026, "partition": PARTITION, "model": {"num_labels": 77, "max_length": 2},
                      "data": {"prepared_root": str(root), "raw_dir": str(root / "raw")},
                      "training": {"eval_max_samples_per_client": None,
                                   "evaluation_splits": ["train", "validation"]}}
            with patch("fl_exp.trainer.verify_prepared"):
                data = ClientData(config, tokenize)
            self.assertFalse(data.dynamic_padding)
            self.assertTrue(all(set(client) == {"train", "validation"} for client in data.clients))
            self.assertTrue(all("test" not in entry["sample_ids"] for entry in data.evaluation_manifest()))

            class ConstantClassifier(torch.nn.Module):
                def forward(self, input_ids, attention_mask):
                    return SimpleNamespace(logits=torch.zeros(len(input_ids), 77))

            result = evaluate(ConstantClassifier(), data, "validation", torch.device("cpu"), 4)
            self.assertEqual(result["evaluation_scope"], "full")
            self.assertEqual(sum(row["n_samples"] for row in result["clients"]), 55)
            self.assertAlmostEqual(result["metrics"]["mean_loss"], math.log(77), places=5)
            self.assertEqual(result["metrics"]["mean_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
