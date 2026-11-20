"""Batch-local right-padding trimming preserves examples and global labels."""

import unittest

import torch

from fl_exp.trainer import ClientData


class DynamicPaddingTest(unittest.TestCase):
    def _data(self, lengths, width, dynamic):
        data = ClientData.__new__(ClientData)
        data.dynamic_padding = dynamic
        ids = torch.zeros(len(lengths), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = torch.arange(1, length + 1) + index * 100
            mask[index, :length] = 1
        tensors = {"input_ids": ids, "attention_mask": mask,
                   "labels": torch.arange(len(lengths), dtype=torch.long) + 5}
        data.clients = [{"train": {"tensors": tensors}}]
        return data, tensors

    def test_disabled_keeps_the_fixed_length_batch(self):
        data, source = self._data([7, 19], 64, False)
        indices = torch.tensor([1, 0])
        batch = data.batch(0, "train", indices.tolist(), "cpu")
        for key, value in source.items():
            self.assertTrue(torch.equal(batch[key], value.index_select(0, indices)))
        self.assertEqual(batch["input_ids"].shape, (2, 64))

    def test_rounding_and_upper_bound_preserve_tokens_and_labels(self):
        cases = (([1, 7, 15], 64, 16), ([16, 17, 31], 64, 32),
                 ([33, 48], 64, 48), ([49, 64], 64, 64),
                 ([17, 29], 30, 30), ([1, 8], 8, 8))
        for lengths, width, expected in cases:
            with self.subTest(lengths=lengths, width=width):
                data, source = self._data(lengths, width, True)
                indices = torch.arange(len(lengths) - 1, -1, -1)
                batch = data.batch(0, "train", indices.tolist(), "cpu")
                self.assertEqual(batch["input_ids"].shape, (len(lengths), expected))
                for key in ("input_ids", "attention_mask"):
                    selected = source[key].index_select(0, indices)
                    self.assertTrue(torch.equal(batch[key], selected[:, :expected]))
                    self.assertTrue(batch[key].is_contiguous())
                self.assertTrue(torch.equal(batch["labels"], source["labels"].index_select(0, indices)))
                self.assertTrue(torch.equal(batch["attention_mask"].sum(dim=1),
                                            source["attention_mask"].sum(dim=1).index_select(0, indices)))
                self.assertEqual(source["input_ids"].shape[1], width)


if __name__ == "__main__":
    unittest.main()
