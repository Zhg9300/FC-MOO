"""Five single-global-point client update rules, independently implemented.

Reference: Zhg9300/FC-MOO, revision 0156b34b2ca324e7d9ccc68f6dba93c70fdf1f5d.
qFFL retains qFedAvg's curvature denominator; FedMGDA+ retains normalized
gradients and epsilon-box weights, centered on equal clients for this study.
FairMOO retains the reference's unbiased switching variance, population
variance gradient, fairness-preserving QP, and unit-length final direction.
FC-MOO follows journal-v1 Algorithm 2: two per-client EMA trackers, a capped
fairness multiplier, and an unnormalized direction. Its beta is the weight of
the NEW observation, unlike the reference's momentum notation. Loss and
gradient betas can be overridden independently; legacy configs share beta.
"""

import numpy as np
import torch

from .solver import solve_client_qp


METHODS = ("fedavg", "qffl", "fedmgda", "fairmoo", "fcmoo")


class FederatedAlgorithm:
    def __init__(self, name, config, n_clients=10):
        if name not in METHODS:
            raise ValueError("Unknown method: " + name)
        self.name, self.config, self.n_clients = name, dict(config), n_clients
        if n_clients < 2:
            raise ValueError("At least two clients are required")
        self.loss_tracker = None
        self.gradient_tracker = None
        self.last_round = 0

    def _qp(self, gram, **kwargs):
        return solve_client_qp(gram, tol=float(self.config.get("qp_tol", 1e-8)),
                               max_iter=int(self.config.get("qp_max_iter", 1000)), **kwargs)

    def direction(self, losses, gradients, round_index, lr):
        """Return a CPU FP32 direction; caller applies w -= lr * direction."""
        losses = np.asarray(losses, dtype=np.float64)
        if (losses.shape != (self.n_clients,) or not np.isfinite(losses).all()
                or np.any(losses < 0) or gradients.ndim != 2 or gradients.shape[0] != self.n_clients):
            raise ValueError("Expected finite nonnegative client losses and an [n,d] gradient matrix")
        if gradients.device.type != "cpu" or gradients.dtype != torch.float32 or not torch.isfinite(gradients).all():
            raise ValueError("Client gradients must be finite CPU FP32 tensors")
        if round_index != self.last_round + 1 or lr <= 0 or not np.isfinite(lr):
            raise ValueError("Rounds must be consecutive from 1 and learning rate must be positive")
        diagnostics = {"client_loss_variance": float(np.var(losses))}
        n = self.n_clients
        if self.name == "fedavg":
            result = gradients.mean(dim=0)
        elif self.name == "qffl":
            q = float(self.config.get("q", 0.1))
            if q < 0:
                raise ValueError("q must be nonnegative")
            safe_losses = np.maximum(losses, 1e-10)
            weights = safe_losses ** q
            norm2 = torch.sum(gradients.double().square(), dim=1).numpy()
            denominator = float(weights.sum() + lr * q * np.sum(safe_losses ** (q - 1) * norm2))
            result = torch.as_tensor(weights / denominator, dtype=torch.float32) @ gradients
            diagnostics.update(q=q, qffl_denominator=denominator)
        else:
            task_gradients = gradients
            if self.name == "fcmoo":
                beta = float(self.config.get("tracking_beta", 0.2))
                loss_beta = float(self.config.get("tracking_beta_loss", beta))
                grad_beta = float(self.config.get("tracking_beta_grad", beta))
                power = float(self.config.get("tracking_power", 0.6))
                offset = float(self.config.get("tracking_offset", 10.0))
                if (any(not np.isfinite(b) or not 0 < b <= 1 for b in (beta, loss_beta, grad_beta))
                        or not np.isfinite(power) or power < 0
                        or not np.isfinite(offset) or offset <= 0):
                    raise ValueError("Invalid tracking schedule")
                factor = (1.0 + (round_index - 1) / offset) ** -power
                beta *= factor
                loss_beta *= factor
                grad_beta *= factor
                if self.loss_tracker is None:
                    # Initialize with the first observation rather than zero.
                    self.loss_tracker = losses.copy()
                    self.gradient_tracker = gradients.clone()
                    effective_beta = 1.0
                else:
                    self.loss_tracker *= 1 - loss_beta
                    self.loss_tracker += loss_beta * losses
                    self.gradient_tracker.mul_(1 - grad_beta).add_(gradients, alpha=grad_beta)
                    effective_beta = beta
                losses, task_gradients = self.loss_tracker, self.gradient_tracker
                diagnostics.update(tracking_beta=beta, tracking_effective_beta=effective_beta,
                                   tracking_beta_loss=loss_beta, tracking_beta_grad=grad_beta,
                                   tracking_effective_beta_loss=1.0 if round_index == 1 else loss_beta,
                                   tracking_effective_beta_grad=1.0 if round_index == 1 else grad_beta,
                                   tracked_loss_variance=float(np.var(losses)))
            # Gram matrices are tiny; accumulation uses float64 on the CPU.
            values = task_gradients.numpy().astype(np.float64)
            gram = values @ values.T
            if self.name == "fedmgda":
                norms = np.sqrt(np.maximum(np.diag(gram), 0))
                inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
                normalized_gram = gram * inverse[:, None] * inverse[None, :]
                epsilon = float(self.config.get("epsilon", 0.1))
                if epsilon < 0:
                    raise ValueError("FedMGDA epsilon must be nonnegative")
                if epsilon == 0:
                    weights = np.full(n, 1 / n)
                else:
                    weights, info = self._qp(normalized_gram,
                        lower=np.full(n, max(0.0, 1 / n - epsilon)),
                        upper=np.full(n, min(1.0, 1 / n + epsilon)))
                    diagnostics.update(info)
                result = torch.as_tensor(weights * inverse, dtype=torch.float32) @ gradients
                diagnostics["client_weights"] = weights.tolist()
            else:
                delta = float(self.config.get("delta", 0.001))
                if delta < 0:
                    raise ValueError("Fairness budget delta must be nonnegative")
                centered = losses - losses.mean()
                fair_coefficients = 2.0 / n * centered
                fair_gram = gram @ fair_coefficients
                variance = float(np.mean(centered ** 2))
                if self.name == "fairmoo":
                    threshold = float(self.config.get("fairmoo_prefer", delta))
                    if threshold < 0:
                        raise ValueError("FairMOO switching threshold must be nonnegative")
                    switching_variance = float(np.var(losses, ddof=1))
                    if switching_variance > threshold:
                        coefficients = fair_coefficients
                        diagnostics["fairmoo_branch"] = "fairness_descent"
                    else:
                        coefficients, info = self._qp(gram, halfspace=fair_gram)
                        diagnostics.update(info)
                        diagnostics["fairmoo_branch"] = "fairness_preserving_mgda"
                    result = torch.as_tensor(coefficients, dtype=torch.float32) @ gradients
                    norm = torch.linalg.vector_norm(result)
                    if norm > 0:
                        result = result / norm
                    diagnostics.update(fairmoo_switch_variance=switching_variance,
                                       fairmoo_prefer=threshold)
                else:
                    rho = float(self.config.get("rho", 5.0))
                    cap = float(self.config.get("mu_cap", 1000.0))
                    if rho <= 0 or cap < 0:
                        raise ValueError("FC-MOO requires rho > 0 and mu_cap >= 0")
                    constraint = variance - delta
                    if cap == 0:
                        weights, info = self._qp(gram)
                        multiplier = 0.0
                    else:
                        augmented = np.empty((n + 1, n + 1))
                        augmented[:n, :n] = gram
                        augmented[:n, n] = augmented[n, :n] = fair_gram
                        augmented[n, n] = fair_coefficients @ fair_gram
                        linear = np.zeros(n + 1)
                        linear[-1] = -rho * constraint
                        solution, info = self._qp(augmented, linear=linear, simplex_dim=n,
                            lower=np.zeros(n + 1), upper=np.r_[np.ones(n), cap])
                        weights, multiplier = solution[:n], float(solution[-1])
                    coefficients = weights + multiplier * fair_coefficients
                    result = torch.as_tensor(coefficients, dtype=torch.float32) @ task_gradients
                    diagnostics.update(info)
                    diagnostics.update(client_weights=weights.tolist(), fairness_multiplier=multiplier,
                        tracked_constraint=constraint, tracked_constraint_violation=max(constraint, 0.0),
                        multiplier_at_cap=bool(cap > 0 and multiplier >= cap - 1e-6))
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite global update direction")
        self.last_round = round_index
        diagnostics["direction_l2"] = float(torch.linalg.vector_norm(result))
        return result, diagnostics

    def state_dict(self):
        return {"name": self.name, "n_clients": self.n_clients, "last_round": self.last_round,
                "loss_tracker": None if self.loss_tracker is None else self.loss_tracker.copy(),
                "gradient_tracker": None if self.gradient_tracker is None else self.gradient_tracker.clone()}

    def load_state_dict(self, state):
        if state["name"] != self.name or state["n_clients"] != self.n_clients:
            raise ValueError("Algorithm checkpoint belongs to another method/client count")
        self.last_round = int(state["last_round"])
        self.loss_tracker = None if state["loss_tracker"] is None else np.asarray(state["loss_tracker"]).copy()
        self.gradient_tracker = None if state["gradient_tracker"] is None else state["gradient_tracker"].cpu().float().clone()
