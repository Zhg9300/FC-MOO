"""Full-data expansion keeps the existing holdouts and client assignments."""

from collections import Counter
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from fl_exp.data import (
    BASE_PARTITIONS, SPLITS, expand_pools, load_raw, make_full_partition,
    normalized_text, validate_clients,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/prepared/banking77/seed_2026"


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FullDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = PREPARED / "manifest.json"
        if not manifest.exists():
            raise unittest.SkipTest("Requires the prepared Banking77 reference data")
        if json.loads(manifest.read_text())["partitions"] != list(BASE_PARTITIONS):
            raise unittest.SkipTest("Requires Banking77 reference data with both default Dirichlet partitions")
        _, cls.raw = load_raw(ROOT / "data/raw/banking77")
        cls.reference = {split: records(PREPARED / "pools" / (split + ".jsonl"))
                         for split in SPLITS}
        cls.full, cls.audit = expand_pools(cls.raw, cls.reference)
        cls.partitions = {}
        for name in BASE_PARTITIONS:
            old_clients = [{split: records(PREPARED / name / "clients" /
                                          ("client_{:02d}".format(client)) / (split + ".jsonl"))
                            for split in SPLITS} for client in range(10)]
            old_metadata = json.loads((PREPARED / name / "partition.json").read_text())
            full_clients, metadata = make_full_partition(cls.full, 2026, old_clients, old_metadata)
            cls.partitions[name] = old_clients, old_metadata, full_clients, metadata

    def test_full_pool_covers_every_clean_candidate_once_without_moving_holdouts(self):
        audit = json.loads((PREPARED / "sampling_audit.json").read_text())
        excluded = {row["id"] for row in audit["excluded_training_rows"]}
        eligible = {row["id"]: row for row in self.raw["train"] if row["id"] not in excluded}
        held_out = {row["id"] for row in self.reference["validation"]}
        observed = {row["id"]: row for row in self.full["train"]}
        self.assertEqual(observed, {key: row for key, row in eligible.items() if key not in held_out})
        self.assertEqual(len(observed), len(self.full["train"]))
        self.assertEqual({split: len(rows) for split, rows in self.full.items()},
                         {"train": 9222, "validation": 770, "test": 3080})
        self.assertEqual(self.full["train"][:3803], self.reference["train"])
        self.assertEqual(self.full["validation"], self.reference["validation"])
        self.assertEqual(self.full["test"], self.raw["test"])
        text_sets = {split: {normalized_text(row["text"]) for row in rows}
                     for split, rows in self.full.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(text_sets[left] & text_sets[right])

    def test_old_training_ownership_and_each_client_holdout_are_preserved(self):
        for name, (old, _, full, metadata) in self.partitions.items():
            with self.subTest(partition=name):
                self.assertEqual(metadata["name"], name + "_full")
                for before, after in zip(old, full):
                    self.assertEqual(after["train"][:len(before["train"])], before["train"])
                    self.assertEqual(after["validation"], before["validation"])
                    self.assertEqual(after["test"], before["test"])
                for split in SPLITS:
                    observed = Counter(row["id"] for client in full for row in client[split])
                    self.assertEqual(observed, Counter(row["id"] for row in self.full[split]))
                validate_clients(full, self.full, name + "_full")

    def test_dirichlet_applies_original_probabilities_to_only_new_samples(self):
        old, old_metadata, clients, metadata = self.partitions["dirichlet_alpha_0.1"]
        self.assertEqual(metadata["client_given_label_probabilities"],
                         old_metadata["client_given_label_probabilities"])
        self.assertEqual(metadata["alpha"], old_metadata["alpha"])
        self.assertEqual(metadata["attempts"], old_metadata["attempts"])
        probabilities = np.asarray(old_metadata["client_given_label_probabilities"])
        for label in range(77):
            before = np.asarray([sum(row["label"] == label for row in client["train"]) for client in old])
            after = np.asarray([sum(row["label"] == label for row in client["train"]) for client in clients])
            added = after - before
            targets = probabilities[label] * added.sum()
            expected = np.floor(targets).astype(int)
            order = sorted(range(10), key=lambda i: (-(targets[i] - expected[i]), i))
            expected[order[:int(added.sum() - expected.sum())]] += 1
            np.testing.assert_array_equal(added, expected)
            np.testing.assert_array_equal(after, np.asarray(metadata["counts_by_label_then_client"]["train"])[label])

    def test_expansion_is_deterministic_and_does_not_mutate_reference(self):
        full, audit = expand_pools(self.raw, self.reference)
        self.assertEqual((full, audit), (self.full, self.audit))
        for name, (old, old_metadata, clients, metadata) in self.partitions.items():
            self.assertEqual(make_full_partition(self.full, 2026, old, old_metadata), (clients, metadata))
            for client, before in enumerate(old):
                for split in SPLITS:
                    original = PREPARED / name / "clients" / ("client_{:02d}".format(client)) / (split + ".jsonl")
                    self.assertEqual(before[split], records(original))

    def test_malformed_reference_cannot_change_or_leak_holdout_samples(self):
        for case in ("overlap", "changed_content", "excluded_training_record"):
            with self.subTest(case=case):
                bad = copy.deepcopy(self.reference)
                if case == "overlap":
                    bad["validation"][0] = bad["train"][0]
                elif case == "changed_content":
                    bad["validation"][0]["text"] = "altered reference text"
                else:
                    audit = json.loads((PREPARED / "sampling_audit.json").read_text())
                    excluded_id = audit["excluded_training_rows"][0]["id"]
                    bad["train"][0] = next(row for row in self.raw["train"] if row["id"] == excluded_id)
                with self.assertRaises(ValueError):
                    expand_pools(self.raw, bad)


if __name__ == "__main__":
    unittest.main()
