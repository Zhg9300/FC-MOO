import unittest

from fl_exp.metrics import summarize_clients


class ClientMetricsTest(unittest.TestCase):
    def test_client_equal_population_metrics(self):
        result = summarize_clients([
            {"client_id": 0, "n_samples": 1, "accuracy": 0.2, "loss": 1.0},
            {"client_id": 1, "n_samples": 99, "accuracy": 0.8, "loss": 3.0},
        ])
        expected = {"mean_accuracy": 0.5, "worst_accuracy": 0.2,
                    "accuracy_variance": 0.09, "mean_loss": 2.0,
                    "worst_loss": 3.0, "loss_variance": 1.0}
        self.assertEqual(set(result), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value)

    def test_invalid_or_empty_evaluation_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_clients([])
        for key, value in (("accuracy", 80), ("loss", float("nan")), ("n_samples", 0)):
            row = {"client_id": 0, "n_samples": 1, "accuracy": 0.8, "loss": 1.0}
            row[key] = value
            with self.assertRaises(ValueError):
                summarize_clients([row])


if __name__ == "__main__":
    unittest.main()
