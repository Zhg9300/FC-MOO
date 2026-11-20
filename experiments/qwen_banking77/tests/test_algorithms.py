"""Analytic small problems; run with python -m unittest discover -s tests."""

import unittest

import numpy as np
import torch

from fl_exp.algorithms import FederatedAlgorithm
from fl_exp.solver import solve_client_qp


class ClientQPTests(unittest.TestCase):
    def test_simplex_known_solution(self):
        # Orthogonal gradients [2,0], [0,1] give lambda=(1/5,4/5).
        weights, info = solve_client_qp(np.diag([4.0, 1.0]))
        np.testing.assert_allclose(weights, [0.2, 0.8], atol=1e-7)
        self.assertLess(info["qp_kkt_residual"], 1e-7)

    def test_box_boundary(self):
        weights, _ = solve_client_qp(np.diag([4.0, 1.0]),
            lower=np.array([0.4, 0.4]), upper=np.array([0.6, 0.6]))
        np.testing.assert_allclose(weights, [0.4, 0.6], atol=1e-7)

    def test_fairness_preserving_halfspace(self):
        weights, _ = solve_client_qp(np.diag([4.0, 1.0]), halfspace=np.array([1.0, -1.0]))
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-7)
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            solve_client_qp(np.eye(2), halfspace=np.array([-1.0, -2.0]))

    def test_singular_gram_zero_direction(self):
        weights, _ = solve_client_qp(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-7)

    def test_fc_multiplier_and_cap(self):
        # One simplex variable is fixed at 1. min .5*(1+mu)^2-3*mu.
        for cap, expected in [(10.0, 2.0), (0.5, 0.5)]:
            solution, _ = solve_client_qp(np.ones((2, 2)), linear=np.array([0.0, -3.0]),
                simplex_dim=1, lower=np.zeros(2), upper=np.array([1.0, cap]))
            np.testing.assert_allclose(solution, [1.0, expected], atol=1e-6)


class AlgorithmTests(unittest.TestCase):
    def test_fedavg_and_q_zero(self):
        gradients = torch.tensor([[1., 3.], [3., -1.]])
        for method in ("fedavg", "qffl"):
            direction, _ = FederatedAlgorithm(method, {"q": 0}, 2).direction([1, 3], gradients, 1, 0.1)
            torch.testing.assert_close(direction, torch.tensor([2., 1.]))

    def test_qffl_curvature_denominator(self):
        direction, _ = FederatedAlgorithm("qffl", {"q": 1}, 2).direction(
            [1, 3], torch.tensor([[1., 0.], [0., 2.]]), 1, 0.2)
        # Weighted numerator [1,6], denominator 1+3+.2*(1+4)=5.
        torch.testing.assert_close(direction, torch.tensor([0.2, 1.2]))

    def test_fedmgda_normalizes_and_uses_equal_center(self):
        direction, _ = FederatedAlgorithm("fedmgda", {"epsilon": 0}, 2).direction(
            [1, 2], torch.tensor([[10., 0.], [0., 2.]]), 1, 0.1)
        torch.testing.assert_close(direction, torch.tensor([0.5, 0.5]))

    def test_fairmoo_switch_and_unit_direction(self):
        direction, info = FederatedAlgorithm("fairmoo", {"delta": 0.1}, 2).direction(
            [1, 3], torch.tensor([[1., 0.], [0., 1.]]), 1, 0.1)
        torch.testing.assert_close(direction, torch.tensor([-1., 1.]) / np.sqrt(2))
        self.assertEqual(info["fairmoo_branch"], "fairness_descent")
        direction, info = FederatedAlgorithm("fairmoo", {"delta": 3}, 2).direction(
            [1, 3], torch.tensor([[1., 0.], [0., 1.]]), 1, 0.1)
        self.assertEqual(info["fairmoo_branch"], "fairness_preserving_mgda")
        self.assertGreaterEqual(float(direction @ torch.tensor([-1., 1.])), -1e-6)

    def test_fc_tracking_schedule_resume_and_feasible_budget(self):
        config = {"delta": 100, "tracking_beta": 0.25, "tracking_power": 0.0,
                  "tracking_offset": 10, "rho": 5, "mu_cap": 1000}
        first = FederatedAlgorithm("fcmoo", config, 2)
        first.direction([1, 3], torch.tensor([[1., 0.], [0., 1.]]), 1, 0.1)
        restored = FederatedAlgorithm("fcmoo", config, 2)
        restored.load_state_dict(first.state_dict())
        gradients = torch.tensor([[3., 0.], [0., 5.]])
        expected, info = first.direction([5, 7], gradients, 2, 0.1)
        actual, _ = restored.direction([5, 7], gradients, 2, 0.1)
        np.testing.assert_allclose(first.loss_tracker, [2, 4])
        torch.testing.assert_close(first.gradient_tracker, torch.tensor([[1.5, 0.], [0., 2.]]))
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(info["fairness_multiplier"], 0, places=6)

    def test_fc_separate_trackers_and_resume(self):
        config = {"delta": 100, "rho": 5, "tracking_beta": 0.1,
                  "tracking_beta_loss": 0.05, "tracking_beta_grad": 0.2,
                  "tracking_power": 0, "tracking_offset": 10}
        first = FederatedAlgorithm("fcmoo", config, 2)
        gradients = torch.tensor([[1., 0.], [0., 1.]])
        _, info = first.direction([1, 3], gradients, 1, 0.1)
        np.testing.assert_array_equal(first.loss_tracker, [1, 3])
        torch.testing.assert_close(first.gradient_tracker, gradients, rtol=0, atol=0)
        self.assertEqual(info["tracking_effective_beta_loss"], 1)
        self.assertEqual(info["tracking_effective_beta_grad"], 1)
        first.direction([5, 7], torch.tensor([[3., 0.], [0., 5.]]), 2, 0.1)
        np.testing.assert_allclose(first.loss_tracker, [1.2, 3.2])
        torch.testing.assert_close(first.gradient_tracker, torch.tensor([[1.4, 0.], [0., 1.8]]))
        restored = FederatedAlgorithm("fcmoo", config, 2)
        restored.load_state_dict(first.state_dict())
        gradients = torch.tensor([[7., 0.], [0., 9.]])
        expected, info = first.direction([9, 11], gradients, 3, 0.1)
        actual, resumed_info = restored.direction([9, 11], gradients, 3, 0.1)
        np.testing.assert_allclose(first.loss_tracker, [1.59, 3.59])
        torch.testing.assert_close(first.gradient_tracker, torch.tensor([[2.52, 0.], [0., 3.24]]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(restored.loss_tracker, first.loss_tracker)
        torch.testing.assert_close(restored.gradient_tracker, first.gradient_tracker, rtol=0, atol=0)
        self.assertEqual(info, resumed_info)
        self.assertEqual(info["tracking_beta_loss"], 0.05)
        self.assertEqual(info["tracking_beta_grad"], 0.2)
        self.assertEqual(info["tracking_effective_beta_loss"], 0.05)
        self.assertEqual(info["tracking_effective_beta_grad"], 0.2)
        self.assertEqual(info["tracking_beta"], 0.1)

    def test_fc_equal_tracker_overrides_preserve_legacy(self):
        for power in [0, 0.2]:
            with self.subTest(power=power):
                config = {"delta": 0.01, "rho": 5, "tracking_beta": 0.2,
                          "tracking_power": power, "tracking_offset": 10}
                legacy = FederatedAlgorithm("fcmoo", config, 2)
                explicit = FederatedAlgorithm("fcmoo", dict(config,
                    tracking_beta_loss=0.2, tracking_beta_grad=0.2), 2)
                for step in range(1, 4):
                    losses = [1 + step, 4 - step * 0.5]
                    gradients = torch.tensor([[float(step), 1.], [-0.5, 2. + step]])
                    expected, old_info = legacy.direction(losses, gradients, step, 0.1)
                    actual, info = explicit.direction(losses, gradients, step, 0.1)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    np.testing.assert_array_equal(explicit.loss_tracker, legacy.loss_tracker)
                    torch.testing.assert_close(explicit.gradient_tracker, legacy.gradient_tracker, rtol=0, atol=0)
                    self.assertEqual(info, old_info)
                    beta = 0.2 * (1 + (step - 1) / 10) ** -power
                    self.assertAlmostEqual(info["tracking_beta_loss"], beta)
                    self.assertAlmostEqual(info["tracking_beta_grad"], beta)

    def test_fc_single_tracker_override_falls_back_independently(self):
        for key, loss_beta, grad_beta in [("tracking_beta_loss", 0.05, 0.2),
                                          ("tracking_beta_grad", 0.2, 0.05)]:
            with self.subTest(key=key):
                config = {"delta": 100, "rho": 5, "tracking_beta": 0.2,
                          "tracking_power": 0, key: 0.05}
                algorithm = FederatedAlgorithm("fcmoo", config, 2)
                algorithm.direction([1, 3], torch.eye(2), 1, 0.1)
                _, info = algorithm.direction([5, 7], 3 * torch.eye(2), 2, 0.1)
                np.testing.assert_allclose(algorithm.loss_tracker, np.array([1, 3]) + 4 * loss_beta)
                torch.testing.assert_close(algorithm.gradient_tracker, (1 + 2 * grad_beta) * torch.eye(2))
                self.assertEqual(info["tracking_beta_loss"], loss_beta)
                self.assertEqual(info["tracking_beta_grad"], grad_beta)

    def test_fc_invalid_separate_tracking_beta(self):
        for key in ["tracking_beta_loss", "tracking_beta_grad"]:
            for value in [0, 1.1, float("nan"), float("inf")]:
                with self.subTest(key=key, value=value):
                    algorithm = FederatedAlgorithm("fcmoo", {key: value}, 2)
                    with self.assertRaises(ValueError):
                        algorithm.direction([1, 3], torch.eye(2), 1, 0.1)

    def test_fc_violated_budget_adds_variance_correction(self):
        direction, info = FederatedAlgorithm("fcmoo", {"delta": 0, "rho": 2}, 2).direction(
            [1, 3], torch.eye(2), 1, 0.1)
        # Variance gradient is [-1,1]; rho*G=2, so its inner product
        # with the optimal update is 2 when the multiplier is not capped.
        self.assertGreater(info["fairness_multiplier"], 0)
        self.assertAlmostEqual(float(direction @ torch.tensor([-1., 1.])), 2, places=5)


if __name__ == "__main__":
    unittest.main()
