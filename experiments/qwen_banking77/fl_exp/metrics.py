"""Client-equal accuracy and cross-entropy metrics."""

import math
from statistics import fmean, pvariance


def summarize_clients(rows):
    """Aggregate client means; client sample counts do not set client weights."""
    if not rows:
        raise ValueError("Cannot summarize an empty client list")
    accuracies = [float(row["accuracy"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in accuracies):
        raise ValueError("Client accuracies must be finite fractions in [0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in losses):
        raise ValueError("Client cross-entropies must be finite and nonnegative")
    if any(row["n_samples"] <= 0 for row in rows):
        raise ValueError("Each evaluated client must have samples")
    return {
        "mean_accuracy": fmean(accuracies),
        "worst_accuracy": min(accuracies),
        "accuracy_variance": pvariance(accuracies),
        "mean_loss": fmean(losses),
        "worst_loss": max(losses),
        "loss_variance": pvariance(losses),
    }
