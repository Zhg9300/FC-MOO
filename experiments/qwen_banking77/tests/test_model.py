"""Tiny CPU checks for classifier initialization; no pretrained model load."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from transformers import Qwen2Config, Qwen2ForSequenceClassification

from fl_exp.model import build_model
from run import validate


class HeadInitializationTest(unittest.TestCase):
    def _build(self, path, scale=None):
        config = {"model_path": path, "num_labels": 3, "lora_r": 2,
                  "lora_alpha": 4, "gradient_checkpointing": False}
        if scale is not None:
            config["head_init_scale"] = scale
        tiny = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=1, num_labels=3, pad_token_id=0)
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        torch.manual_seed(2026)
        with patch("fl_exp.model.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("fl_exp.model.AutoModelForSequenceClassification.from_pretrained",
                   # Mimic BF16 source weights so scaling before the FP32 cast
                   # would lose precision and fail the exact scaling assertion.
                   side_effect=lambda *args, **kwargs: Qwen2ForSequenceClassification(tiny).bfloat16()):
            model, _ = build_model(config, "cpu")
        return model, torch.get_rng_state().clone()

    def test_default_and_explicit_one_are_identical(self):
        with tempfile.TemporaryDirectory() as path:
            old, old_rng = self._build(path)
            explicit, explicit_rng = self._build(path, 1.0)
        self.assertEqual(list(old.state_dict()), list(explicit.state_dict()))
        for name, tensor in old.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[name]), name)
        self.assertTrue(torch.equal(old_rng, explicit_rng))

    def test_point_one_only_scales_active_fp32_head(self):
        with tempfile.TemporaryDirectory() as path:
            baseline, baseline_rng = self._build(path)
            scaled, scaled_rng = self._build(path, 0.1)
        parameters = dict(baseline.named_parameters())
        head_names = [name for name, parameter in parameters.items()
                      if parameter.requires_grad and "score" in name]
        self.assertEqual(len(head_names), 1)
        head_name = head_names[0]
        for name, parameter in scaled.named_parameters():
            if name == head_name:
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertTrue(torch.equal(parameter, parameters[name] * 0.1))
            else:
                self.assertTrue(torch.equal(parameter, parameters[name]), name)
        self.assertTrue(torch.equal(baseline_rng, scaled_rng))

    def test_invalid_scale_rejected_by_model_and_launcher(self):
        root = Path(__file__).resolve().parents[1]
        template = yaml.safe_load((root / "configs/default.yaml").read_text())
        for scale in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "head_init_scale"):
                    build_model({"head_init_scale": scale}, "cpu")
                config = copy.deepcopy(template)
                config["model"]["head_init_scale"] = scale
                with self.assertRaisesRegex(ValueError, "head_init_scale"):
                    validate(config)
        config = copy.deepcopy(template)
        config["model"].pop("head_init_scale", None)
        validate(config)
        self.assertNotIn("head_init_scale", config["model"])


if __name__ == "__main__":
    unittest.main()
